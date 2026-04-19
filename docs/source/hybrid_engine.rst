Hybrid Engine
=============

The **Hybrid Engine** colocates all RL components — Actor, Critic, Reward, Reference, and the
vLLM engines — on the **same** GPUs and time-slices them via memory sleep mode. It is the
recommended setup when GPU memory allows: it gives the simplest deployment, the lowest GPU count,
and typically the best throughput.

For the broader architecture see :doc:`architecture`; for tuning see :doc:`performance`.

.. contents::
   :local:
   :depth: 2

How sleep mode works
--------------------

In a naive distributed RL setup, vLLM idles during the training phase and DeepSpeed idles during
generation — wasting half the GPU clock. Hybrid Engine fixes this by **time-sharing the same
GPUs**:

1. **Generation phase**: vLLM is awake and uses most of the GPU (KV cache + weights). DeepSpeed
   engines are *asleep* (offloaded / minimal footprint).
2. **Weight sync**: trainer broadcasts updated actor weights to vLLM via NCCL
   (``--vllm.sync_backend nccl``).
3. **Training phase**: vLLM goes to sleep (``--vllm.enable_sleep``), DeepSpeed wakes
   (``--ds.enable_sleep``) and runs forward + backward + optimizer step on the actor / critic /
   reference / reward.

Because both sides know how to sleep, they can fit on one GPU set even at large model sizes. The
only memory you pay full-time for is whatever each side needs to be *resident* (model weights,
KV cache budget controlled by ``--vllm.gpu_memory_utilization``).

.. _hybrid_engine:

Launch recipe (Qwen3-4B RLVR — math)
------------------------------------

This is the default RL example used throughout the docs: **Qwen3-4B-Thinking** trained with
**REINFORCE++-baseline** on math reasoning (RLVR), using a Python reward function for answer
verification. Adapted from the upstream `train_reinforce_baseline_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_hybrid_engine.sh>`_
and `train_prorlv2_math_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_prorlv2_math_hybrid_engine.sh>`_.

.. code-block:: bash

   # launch the master node of ray in a container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # additional worker nodes (optional)
   ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
      --reward.remote_url examples/python/math_reward_func.py \
      --data.prompt_dataset zhuzilin/dapo-math-17k \
      --data.input_key prompt \
      --data.label_key label \
      --data.apply_chat_template \
      --ds.packing_samples \
      \
      --ref.num_nodes 1 \
      --ref.num_gpus_per_node 4 \
      --actor.num_nodes 1 \
      --actor.num_gpus_per_node 4 \
      --vllm.num_engines 2 \
      --vllm.tensor_parallel_size 2 \
      --train.colocate_all \
      --vllm.gpu_memory_utilization 0.7 \
      --vllm.enable_sleep \
      --ds.enable_sleep \
      --vllm.sync_backend nccl \
      --vllm.enforce_eager \
      \
      --algo.advantage.estimator reinforce_baseline \
      --algo.kl.use_loss \
      --algo.kl.estimator k2 \
      --algo.kl.init_coef 1e-5 \
      --actor.entropy_coef 0.0 \
      --algo.advantage.is_correction_enable \
      --algo.advantage.is_correction_type icepop \
      \
      --rollout.batch_size 128 \
      --rollout.n_samples_per_prompt 8 \
      --train.batch_size 1024 \
      --algo.dynamic_filtering_enable \
      --algo.dynamic_filtering_range 0.0 1.0 \
      --train.dynamic_batch_enable \
      --train.max_tokens_per_gpu 16192 \
      --rollout.max_tokens_per_gpu 32768 \
      --train.micro_batch_size 1 \
      --rollout.micro_batch_size 8 \
      --data.max_len 74240 \
      --rollout.max_new_tokens 64000 \
      --data.max_samples 128000 \
      --train.max_epochs 1 \
      --train.num_episodes 1 \
      \
      --ds.zero_stage 3 \
      --ds.param_dtype bf16 \
      --actor.gradient_checkpointing_enable \
      --ds.ring_attn_size 2 \
      --ds.ring_attn_head_stride 2 \
      --actor.adam.lr 5e-7 \
      \
      --ckpt.output_dir ./exp/Qwen3-4B-Thinking \
      --ckpt.path ./exp/Qwen3-4B-Thinking/ckpt \
      --ckpt.save_hf \
      --ckpt.max_num 3 \
      --ckpt.save_steps 10 \
      --logger.logging_steps 1 \
      --eval.steps -1

.. note::

   - Works with any RL algorithm — change ``--algo.advantage.estimator`` to switch.
   - Works with both single-turn and multi-turn agent modes (see :doc:`agent_training`).
   - To drive the actor with Muon, add ``--actor.optim muon`` (requires DeepSpeed ≥ 0.18.2 and
     drops ``--ds.adam_offload``; see :doc:`common_options`).

Key flags
---------

Essential
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--train.colocate_all``
     - Colocate vLLM engines, Actor, Reference, Reward, and Critic on the same GPUs.
   * - ``--vllm.gpu_memory_utilization <0..1>``
     - vLLM KV-cache fraction. Start at ``0.5`` for 8×A100 and increase if stable.
   * - ``--vllm.enable_sleep``
     - vLLM sleep mode — frees most of vLLM's memory between rollouts.
   * - ``--ds.enable_sleep``
     - DeepSpeed sleep mode — frees DeepSpeed memory between training steps.
   * - ``--vllm.sync_backend nccl``
     - NCCL backend for weight sync (faster than the default).
   * - ``--vllm.enforce_eager``
     - Disable CUDA graphs in vLLM (required for some setups; reduces memory).

Finer-grained colocation (when **not** using ``--train.colocate_all``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--train.colocate_critic_reward``
     - Place Critic and Reward on the same GPUs.
   * - ``--train.colocate_actor_ref``
     - Place Actor and Reference on the same GPUs.
   * - ``--ref.offload`` / ``--reward.offload``
     - Offload Reference / Reward to CPU during the actor's training phase.

Memory rule of thumb
--------------------

``vllm.gpu_memory_utilization`` + model memory < 1.0. Examples for 8×A100 (80GB):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model size
     - Suggested ``--vllm.gpu_memory_utilization``
   * - 8B
     - ``0.6`` (room for full RLHF stack)
   * - 13B
     - ``0.5``
   * - 34B
     - ``0.4`` (consider distributed mode)
   * - 70B+
     - Prefer distributed mode (separate GPU groups per role)

Relationship to async training
------------------------------

``--vllm.enable_sleep`` is **incompatible** with ``--train.async_enable`` (the trainer asserts
this — async mode keeps vLLM running). ``--train.colocate_all`` may still be combined with
``--train.async_enable``, but in async mode it only colocates the **DeepSpeed** models on shared
GPUs; vLLM keeps its own GPU group so it can keep generating. For higher throughput at the cost
of off-policy noise, see :doc:`async_training`.

When **not** to use Hybrid Engine
---------------------------------

Switch to distributed mode (separate GPU groups per role) when:

- You hit OOM even after lowering ``--vllm.gpu_memory_utilization`` and enabling all memory savers.
- You're training models large enough (70B+) that no single GPU can host model + KV cache.
- You want maximum throughput via async + partial rollout (see :doc:`agent_training`).

See :doc:`performance` for the full tuning guide and :doc:`troubleshooting` for OOM / NCCL /
vLLM-hang issues.

Hybrid Engine
=============

The **Hybrid Engine** colocates all RL components — Actor, Critic, Reward, Reference, and the vLLM engines — on the **same** GPUs and time-slices them via memory sleep mode. It is the recommended setup when GPU memory allows: it gives the simplest deployment, the lowest GPU count, and typically the best throughput.

For the broader architecture see :doc:`architecture`; for tuning see :doc:`performance`.

.. contents::
   :local:
   :depth: 2

How sleep mode works
--------------------

In a naive distributed RL setup, vLLM idles during the training phase and DeepSpeed idles during generation — wasting half the GPU clock. Hybrid Engine fixes this by **time-sharing the same GPUs**:

1. **Generation phase**: vLLM is awake and uses most of the GPU (KV cache + weights). DeepSpeed engines are *asleep* (offloaded / minimal footprint).
2. **Weight sync**: trainer broadcasts updated actor weights to vLLM via NCCL (``--vllm_sync_backend nccl``).
3. **Training phase**: vLLM goes to sleep (``--vllm_enable_sleep``), DeepSpeed wakes (``--deepspeed_enable_sleep``) and runs forward + backward + optimizer step on the actor / critic / reference / reward.

Because both sides know how to sleep, they can fit on one GPU set even at large model sizes. The only memory you pay full-time for is whatever each side needs to be *resident* (model weights, KV cache budget controlled by ``--vllm_gpu_memory_utilization``).

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
      --pretrain Qwen/Qwen3-4B-Thinking-2507 \
      --remote_rm_url examples/python/math_reward_func.py \
      --prompt_data zhuzilin/dapo-math-17k \
      --input_key prompt \
      --label_key label \
      --apply_chat_template \
      --packing_samples \
      \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 4 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.7 \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep \
      --vllm_sync_backend nccl \
      --enforce_eager \
      \
      --advantage_estimator reinforce_baseline \
      --use_kl_loss \
      --kl_estimator k2 \
      --init_kl_coef 1e-5 \
      --entropy_loss_coef 0.0 \
      --enable_vllm_is_correction \
      --vllm_is_correction_type icepop \
      \
      --rollout_batch_size 128 \
      --n_samples_per_prompt 8 \
      --train_batch_size 1024 \
      --dynamic_filtering \
      --dynamic_filtering_reward_range 0.0 1.0 \
      --use_dynamic_batch \
      --train_max_tokens_per_gpu 16192 \
      --rollout_max_tokens_per_gpu 32768 \
      --micro_train_batch_size 1 \
      --micro_rollout_batch_size 8 \
      --max_len 74240 \
      --max_new_tokens 64000 \
      --max_samples 128000 \
      --max_epochs 1 \
      --num_episodes 1 \
      \
      --zero_stage 3 \
      --param_dtype bf16 \
      --gradient_checkpointing \
      --ring_attn_size 2 \
      --ring_head_stride 2 \
      --actor_learning_rate 5e-7 \
      \
      --save_path ./exp/Qwen3-4B-Thinking \
      --ckpt_path ./exp/Qwen3-4B-Thinking/ckpt \
      --save_hf_ckpt \
      --max_ckpt_num 3 \
      --save_steps 10 \
      --logging_steps 1 \
      --eval_steps -1

.. note::

   - Works with any RL algorithm — change ``--advantage_estimator`` to switch.
   - Works with both single-turn and multi-turn agent modes (see :doc:`agent_training`).

Key flags
---------

Essential
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--colocate_all_models``
     - Colocate vLLM engines, Actor, Reference, Reward, and Critic on the same GPUs.
   * - ``--vllm_gpu_memory_utilization <0..1>``
     - vLLM KV-cache fraction. Start at ``0.5`` for 8×A100 and increase if stable.
   * - ``--vllm_enable_sleep``
     - vLLM sleep mode — frees most of vLLM's memory between rollouts.
   * - ``--deepspeed_enable_sleep``
     - DeepSpeed sleep mode — frees DeepSpeed memory between training steps.
   * - ``--vllm_sync_backend nccl``
     - NCCL backend for weight sync (faster than the default).
   * - ``--enforce_eager``
     - Disable CUDA graphs in vLLM (required for some setups; reduces memory).

Finer-grained colocation (when **not** using ``--colocate_all_models``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--colocate_critic_reward``
     - Place Critic and Reward on the same GPUs.
   * - ``--colocate_actor_ref``
     - Place Actor and Reference on the same GPUs.
   * - ``--ref_reward_offload``
     - Offload Reference and Reward to CPU during the actor's training phase.

Memory rule of thumb
--------------------

``vllm_gpu_memory_utilization`` + model memory < 1.0. Examples for 8×A100 (80GB):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model size
     - Suggested ``--vllm_gpu_memory_utilization``
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

``--vllm_enable_sleep`` is **incompatible** with ``--async_train`` (the trainer asserts this — async mode keeps vLLM running). ``--colocate_all_models`` may still be combined with ``--async_train``, but in async mode it only colocates the **DeepSpeed** models on shared GPUs; vLLM keeps its own GPU group so it can keep generating. For higher throughput at the cost of off-policy noise, see :doc:`async_training`.

When **not** to use Hybrid Engine
---------------------------------

Switch to distributed mode (separate GPU groups per role) when:

- You hit OOM even after lowering ``--vllm_gpu_memory_utilization`` and enabling all memory savers.
- You're training models large enough (70B+) that no single GPU can host model + KV cache.
- You want maximum throughput via async + partial rollout (see :doc:`agent_training`).

See :doc:`performance` for the full tuning guide and :doc:`troubleshooting` for OOM / NCCL / vLLM-hang issues.

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

Launch recipe
-------------

.. code-block:: bash

   # launch the master node of ray in a container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # additional worker nodes (optional)
   ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 --reward_num_gpus_per_node 8 \
      --actor_num_nodes 1 --actor_num_gpus_per_node 8 \
      --vllm_num_engines 8 --vllm_tensor_parallel_size 1 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.6 \
      --advantage_estimator reinforce_baseline \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 4 --train_batch_size 128 \
      --micro_rollout_batch_size 8 --rollout_batch_size 1024 \
      --n_samples_per_prompt 4 --max_epochs 1 \
      --max_len 2048 --max_samples 20000 \
      --zero_stage 3 --param_dtype bf16 \
      --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
      --init_kl_coef 1e-4 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages --apply_chat_template \
      --normalize_reward --gradient_checkpointing \
      --packing_samples --use_dynamic_batch \
      --vllm_sync_backend nccl --enforce_eager \
      --vllm_enable_sleep --deepspeed_enable_sleep

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

Async training is mutually exclusive
------------------------------------

Do **not** combine ``--async_train`` with ``--colocate_all_models``. The async pipeline overlaps rollout and training across processes — it cannot share GPUs with sleep-mode hybrid execution. For higher throughput at the cost of off-policy noise, see the async + partial-rollout section in :doc:`agent_training`.

When **not** to use Hybrid Engine
---------------------------------

Switch to distributed mode (separate GPU groups per role) when:

- You hit OOM even after lowering ``--vllm_gpu_memory_utilization`` and enabling all memory savers.
- You're training models large enough (70B+) that no single GPU can host model + KV cache.
- You want maximum throughput via async + partial rollout (see :doc:`agent_training`).

See :doc:`performance` for the full tuning guide and :doc:`troubleshooting` for OOM / NCCL / vLLM-hang issues.

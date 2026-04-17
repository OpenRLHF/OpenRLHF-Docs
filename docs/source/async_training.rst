Async Training & Partial Rollout
================================

Sync RLHF (the default) executes one phase at a time: rollout finishes, then training runs, then rollout starts again. **Async training** removes that bubble by running rollout and training concurrently. **Partial rollout** goes one step further by overlapping weight sync with generation itself.

These are throughput-oriented features — they trade a small amount of on-policy-ness for substantially higher GPU utilization. Use them when convergence has been validated in sync mode.

For the underlying architecture see :doc:`architecture`; for the alternative (sync) pipeline see :doc:`hybrid_engine`.

.. contents::
   :local:
   :depth: 2

How it works
------------

**Sync pipeline (default)**::

   ┌────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
   │  rollout   │───▶│  train   │───▶│  rollout   │───▶│  train   │
   └────────────┘    └──────────┘    └────────────┘    └──────────┘
                          (one phase at a time, GPUs alternate roles)

**Async pipeline (** ``--async_train`` **)**::

   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
   │  rollout   │─▶│  rollout   │─▶│  rollout   │─▶│  rollout   │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │ (queue)       │               │               │
         ▼               ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  train   │───▶│  train   │───▶│  train   │───▶│  train   │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
   (rollout and train run concurrently; bounded queue between them)

**Async + Partial Rollout (** ``--partial_rollout`` **)**: vLLM never fully stops. When the trainer pushes new weights, vLLM **pauses** the in-flight requests, swaps weights, and **resumes** — so a single sample can contain tokens generated under both old and new weights. Off-policy noise in exchange for full overlap.

Flags
-----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--async_train``
     - Enable the async pipeline. Rollout and training overlap through a bounded queue.
   * - ``--async_queue_size``
     - Queue depth between rollout and training (default ``1``). Larger values raise throughput but increase off-policy lag.
   * - ``--partial_rollout``
     - Use vLLM pause/resume for weight sync, so generation overlaps with weight broadcast. **Requires** ``--async_train``.
   * - ``--vllm_generate_batch_size``
     - vLLM generation batch size; setting it larger than ``--rollout_batch_size`` enables oversampling. **Requires** ``--async_train`` when greater than ``--rollout_batch_size``.

Compatibility notes:

- ``--async_train`` is **incompatible** with ``--vllm_enable_sleep``. The trainer asserts this; remove ``--vllm_enable_sleep`` before adding ``--async_train``.
- ``--colocate_all_models`` may be combined with ``--async_train`` — but in async mode it only colocates the **DeepSpeed** models (Actor / Ref / Critic / Reward) on shared GPUs; vLLM keeps its own GPU group so it can keep generating.
- For maximum GPU utilization without async, prefer the sync :doc:`hybrid_engine` (``--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep``).

Launch recipe (async + partial rollout)
---------------------------------------

This is the upstream `train_reinforce_baseline_ray_agent_async.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_ flattened into a single ``ray job submit`` invocation:

.. code-block:: bash

   export VLLM_USE_V1=1

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3-4B-Thinking-2507 \
      --agent_func_path examples/python/agent_func.py \
      --prompt_data zhuzilin/dapo-math-17k \
      --input_key prompt \
      --label_key label \
      --apply_chat_template \
      --packing_samples \
      \
      --async_train \
      --partial_rollout \
      \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 4 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 2 \
      --vllm_gpu_memory_utilization 0.95 \
      --colocate_all_models \
      --deepspeed_enable_sleep \
      --vllm_sync_backend nccl \
      --enforce_eager \
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
      --advantage_estimator reinforce_baseline \
      --actor_learning_rate 5e-7 \
      --entropy_loss_coef 0.0 \
      --init_kl_coef 1e-5 \
      --use_kl_loss \
      --kl_estimator k2 \
      --enable_vllm_is_correction \
      --vllm_is_correction_type icepop \
      \
      --zero_stage 3 \
      --gradient_checkpointing \
      --ring_attn_size 2 \
      --ring_head_stride 2 \
      --param_dtype bf16 \
      \
      --save_path ./exp/Qwen3-4B-Thinking \
      --ckpt_path ./exp/Qwen3-4B-Thinking/ckpt \
      --save_hf_ckpt \
      --max_ckpt_num 3 \
      --save_steps 10 \
      --logging_steps 1 \
      --eval_steps -1

Tuning guide
------------

- **Start with ``--async_queue_size 1``** — this is the smallest off-policy lag (~1 step). Increase only if rollout is bottlenecking training.
- **Pair partial rollout with off-policy correction**. Because in-flight samples mix old and new weights, enable ``--enable_vllm_is_correction`` (typically ``--vllm_is_correction_type icepop`` for reasoning workloads).
- **Validate convergence in sync mode first**. Async + partial rollout can mask convergence regressions caused by other knobs.
- **Use bigger ``--vllm_generate_batch_size``** when generation underutilizes vLLM — async mode lets generation oversample without blocking the trainer.

When **not** to use async
-------------------------

- Tasks sensitive to off-policy noise (small models, short rollouts, sparse rewards).
- When convergence in sync mode hasn't been validated yet.
- Single-node small-scale runs where the sync overhead is already low.

In those cases use the :doc:`hybrid_engine` for best throughput-with-stability.

.. warning::
   Async training and partial rollout deliver the highest throughput but can affect convergence on sensitive tasks. Always validate convergence in sync mode before switching.

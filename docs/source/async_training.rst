Async Training & Partial Rollout
================================

Sync RLHF (the default) executes one phase at a time: rollout finishes, then training runs, then
rollout starts again. **Async training** removes that bubble by running rollout and training
concurrently. **Partial rollout** goes one step further by overlapping weight sync with generation
itself.

These are throughput-oriented features — they trade a small amount of on-policy-ness for
substantially higher GPU utilization. Use them when convergence has been validated in sync mode.

For the underlying architecture see :doc:`architecture`; for the alternative (sync) pipeline see
:doc:`hybrid_engine`.

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

**Async pipeline (** ``--train.async_enable`` **)**::

   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
   │  rollout   │─▶│  rollout   │─▶│  rollout   │─▶│  rollout   │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │ (queue)       │               │               │
         ▼               ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  train   │───▶│  train   │───▶│  train   │───▶│  train   │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
   (rollout and train run concurrently; bounded queue between them)

**Async + Partial Rollout (** ``--train.partial_rollout_enable`` **)**: vLLM never fully stops.
When the trainer pushes new weights, vLLM **pauses** the in-flight requests, swaps weights, and
**resumes** — so a single sample can contain tokens generated under both old and new weights.
Off-policy noise in exchange for full overlap.

Flags
-----

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--train.async_enable``
     - Enable the async pipeline. Rollout and training overlap through a bounded queue.
   * - ``--train.async_queue_size``
     - Queue depth between rollout and training (default ``1``). Larger values raise throughput
       but increase off-policy lag.
   * - ``--train.partial_rollout_enable``
     - Use vLLM pause/resume for weight sync, so generation overlaps with weight broadcast.
       **Requires** ``--train.async_enable``.
   * - ``--rollout.vllm_generate_batch_size``
     - vLLM generation batch size; setting it larger than ``--rollout.batch_size`` enables
       oversampling. **Requires** ``--train.async_enable`` when greater than ``--rollout.batch_size``.

Compatibility notes:

- ``--train.async_enable`` is **incompatible** with ``--vllm.enable_sleep``. The trainer asserts
  this; remove ``--vllm.enable_sleep`` before adding ``--train.async_enable``.
- ``--train.colocate_all`` may be combined with ``--train.async_enable`` — but in async mode it
  only colocates the **DeepSpeed** models (Actor / Ref / Critic / Reward) on shared GPUs; vLLM
  keeps its own GPU group so it can keep generating.
- For maximum GPU utilization without async, prefer the sync :doc:`hybrid_engine`
  (``--train.colocate_all --vllm.enable_sleep --ds.enable_sleep``).

Launch recipe (async + partial rollout)
---------------------------------------

This is the upstream `train_reinforce_baseline_ray_agent_async.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_
flattened into a single ``ray job submit`` invocation:

.. code-block:: bash

   export VLLM_USE_V1=1

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
      --train.agent_func_path examples/python/agent_func.py \
      --data.prompt_dataset zhuzilin/dapo-math-17k \
      --data.input_key prompt \
      --data.label_key label \
      --data.apply_chat_template \
      --ds.packing_samples \
      \
      --train.async_enable \
      --train.partial_rollout_enable \
      \
      --ref.num_nodes 1 \
      --ref.num_gpus_per_node 4 \
      --actor.num_nodes 1 \
      --actor.num_gpus_per_node 4 \
      --vllm.num_engines 2 \
      --vllm.tensor_parallel_size 2 \
      --vllm.gpu_memory_utilization 0.95 \
      --train.colocate_all \
      --ds.enable_sleep \
      --vllm.sync_backend nccl \
      --vllm.enforce_eager \
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
      --algo.advantage.estimator reinforce_baseline \
      --actor.adam.lr 5e-7 \
      --actor.entropy_coef 0.0 \
      --algo.kl.init_coef 1e-5 \
      --algo.kl.use_loss \
      --algo.kl.estimator k2 \
      --algo.advantage.is_correction_enable \
      --algo.advantage.is_correction_type icepop \
      \
      --ds.zero_stage 3 \
      --actor.gradient_checkpointing_enable \
      --ds.ring_attn_size 2 \
      --ds.ring_attn_head_stride 2 \
      --ds.param_dtype bf16 \
      \
      --ckpt.output_dir ./exp/Qwen3-4B-Thinking \
      --ckpt.path ./exp/Qwen3-4B-Thinking/ckpt \
      --ckpt.save_hf \
      --ckpt.max_num 3 \
      --ckpt.save_steps 10 \
      --logger.logging_steps 1 \
      --eval.steps -1

Tuning guide
------------

- **Start with ``--train.async_queue_size 1``** — this is the smallest off-policy lag (~1 step).
  Increase only if rollout is bottlenecking training.
- **Pair partial rollout with off-policy correction**. Because in-flight samples mix old and new
  weights, enable ``--algo.advantage.is_correction_enable`` (typically
  ``--algo.advantage.is_correction_type icepop`` for reasoning workloads).
- **Validate convergence in sync mode first**. Async + partial rollout can mask convergence
  regressions caused by other knobs.
- **Use bigger ``--rollout.vllm_generate_batch_size``** when generation underutilizes vLLM —
  async mode lets generation oversample without blocking the trainer.

When **not** to use async
-------------------------

- Tasks sensitive to off-policy noise (small models, short rollouts, sparse rewards).
- When convergence in sync mode hasn't been validated yet.
- Single-node small-scale runs where the sync overhead is already low.

In those cases use the :doc:`hybrid_engine` for best throughput-with-stability.

.. warning::
   Async training and partial rollout deliver the highest throughput but can affect convergence
   on sensitive tasks. Always validate convergence in sync mode before switching.

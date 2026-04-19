Performance Tuning
==================

This guide collects the knobs that matter most for throughput and memory. For hybrid-engine
specifics see :doc:`hybrid_engine`; for error triage see :doc:`troubleshooting`.

**Rule of thumb**: start from a known-good recipe (see ``examples/scripts``) and adjust one knob
at a time.

Choosing a deployment mode
--------------------------

Pick based on what you're optimizing for:

- **Max throughput → async training** (``--train.async_enable``). Rollout and training overlap
  through a bounded queue; tune the degree of asynchrony via ``--train.async_queue_size`` (start
  at ``1`` and raise only if rollout bottlenecks training). Add ``--train.partial_rollout_enable``
  to overlap weight sync with generation. See :doc:`async_training`.
- **Max stability → Hybrid Engine** (``--train.colocate_all --vllm.enable_sleep
  --ds.enable_sleep``). Fully on-policy, excellent GPU utilization through role-swapping on
  shared GPUs. The safe default for convergence-sensitive workloads. See :doc:`hybrid_engine`.
- **Distributed mode** (separate GPU groups for vLLM / Actor / Critic) is a fallback for very
  large models or mixed-hardware clusters where colocation isn't viable. Size each group to its
  own memory and compute needs.

Speed knobs
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 34 41

   * - Knob
     - Flag
     - When / why
   * - **Sample packing**
     - ``--ds.packing_samples``
     - Always on — removes padding, large training speedup.
   * - **NCCL weight sync**
     - ``--vllm.sync_backend nccl``
     - Always on for multi-GPU — faster than the default.
   * - **Dynamic batch**
     - ``--train.dynamic_batch_enable`` + ``--train.max_tokens_per_gpu`` /
       ``--rollout.max_tokens_per_gpu``
     - Variable sequence lengths — better utilization than fixed micro-batches.
   * - **Async training**
     - ``--train.async_enable``
     - Fastest path. Tune degree of asynchrony with ``--train.async_queue_size`` (start at ``1``).
       Validate convergence in sync mode first.
   * - **Partial rollout**
     - ``--train.partial_rollout_enable``
     - With ``--train.async_enable`` — overlap generation with weight sync. Pair with
       ``--algo.advantage.is_correction_enable``.
   * - **Hybrid Engine**
     - ``--train.colocate_all`` + ``--vllm.enable_sleep`` + ``--ds.enable_sleep``
     - Most stable high-throughput option. Best choice when off-policy noise is a concern.
   * - **Overlap comm**
     - ``--ds.overlap_comm``
     - Enough GPU memory — overlap backward and gradient reduce.
   * - **DeepCompile**
     - ``--ds.deepcompile``
     - PyTorch 2.0+ — DeepSpeed graph compilation.
   * - **Prefix caching**
     - ``--vllm.enable_prefix_caching``
     - ``--rollout.n_samples_per_prompt > 1`` — reuse shared prompt KV.

Memory management
-----------------

When memory is **plentiful**:

- Disable ``--ds.adam_offload``; enable ``--ds.overlap_comm``.
- Use ``--train.colocate_all`` (Hybrid Engine), or at least ``--train.colocate_critic_reward`` +
  ``--train.colocate_actor_ref``.

When hitting **OOM** (priority order):

1. Enable ``--ds.packing_samples`` and ``--actor.gradient_checkpointing_enable``.
2. Reduce ``--train.micro_batch_size`` / ``--rollout.micro_batch_size``.
3. Lower ``--vllm.gpu_memory_utilization`` (e.g., 0.6 → 0.5 → 0.4).
4. Enable ``--ds.adam_offload``; raise ``--ds.zero_stage`` (2 → 3).
5. Disable colocation (remove ``--train.colocate_*``) and move to distributed mode.

.. note::
   ``--ds.adam_offload`` is **incompatible** with ``--actor.optim muon`` / ``--critic.optim muon``
   — DeepSpeed's Muon keeps optimizer state on GPU. Use Adam or disable Muon if you need
   adam-offload for memory.

Batch size tuning
-----------------

- **Generation**: maximize ``--rollout.micro_batch_size`` and minimize vLLM TP size (prefer more
  engines over larger TP).
- **Training**: maximize ``--train.micro_batch_size`` with ``--ds.packing_samples`` enabled.
- **Batch relation**: a common choice is
  ``train.batch_size = rollout.batch_size * rollout.n_samples_per_prompt``.

Long context (>8K tokens)
-------------------------

- Enable RingAttention (``--ds.ring_attn_size``) — see :doc:`sequence_parallelism`.
- Keep ``--ds.packing_samples`` on.
- Increase ``--ds.zero_stage`` (typically 3) and watch memory closely.

For the launch recipes see :doc:`hybrid_engine` and :doc:`agent_training`; for error triage see
:doc:`troubleshooting`.

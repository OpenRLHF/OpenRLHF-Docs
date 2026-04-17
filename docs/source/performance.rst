Performance Tuning
==================

This guide collects the knobs that matter most for throughput and memory. For hybrid-engine specifics see :doc:`hybrid_engine`; for error triage see :doc:`troubleshooting`.

**Rule of thumb**: start from a known-good recipe (see ``examples/scripts``) and adjust one knob at a time.

Choosing a deployment mode
--------------------------

Pick based on what you're optimizing for:

- **Max throughput → async training** (``--async_train``). Rollout and training overlap through a bounded queue; tune the degree of asynchrony via ``--async_queue_size`` (start at ``1`` and raise only if rollout bottlenecks training). Add ``--partial_rollout`` to overlap weight sync with generation. See :doc:`async_training`.
- **Max stability → Hybrid Engine** (``--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep``). Fully on-policy, excellent GPU utilization through role-swapping on shared GPUs. The safe default for convergence-sensitive workloads. See :doc:`hybrid_engine`.
- **Distributed mode** (separate GPU groups for vLLM / Actor / Critic) is a fallback for very large models or mixed-hardware clusters where colocation isn't viable. Size each group to its own memory and compute needs.

Speed knobs
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Knob
     - Flag
     - When / why
   * - **Sample packing**
     - ``--packing_samples``
     - Always on — removes padding, large training speedup.
   * - **NCCL weight sync**
     - ``--vllm_sync_backend nccl``
     - Always on for multi-GPU — faster than the default.
   * - **Dynamic batch**
     - ``--use_dynamic_batch`` + ``--train_max_tokens_per_gpu`` / ``--rollout_max_tokens_per_gpu``
     - Variable sequence lengths — better utilization than fixed micro-batches.
   * - **Async training**
     - ``--async_train``
     - Fastest path. Tune the degree of asynchrony with ``--async_queue_size`` (start at ``1``). Validate convergence in sync mode first.
   * - **Partial rollout**
     - ``--partial_rollout``
     - With ``--async_train`` — overlap generation with weight sync. Pair with ``--enable_vllm_is_correction``.
   * - **Hybrid Engine**
     - ``--colocate_all_models`` + ``--vllm_enable_sleep`` + ``--deepspeed_enable_sleep``
     - Most stable high-throughput option. Best choice when off-policy noise is a concern.
   * - **Overlap comm**
     - ``--overlap_comm``
     - Enough GPU memory — overlap backward and gradient reduce.
   * - **DeepCompile**
     - ``--deepcompile``
     - PyTorch 2.0+ — DeepSpeed graph compilation.
   * - **Prefix caching**
     - vLLM config
     - ``--n_samples_per_prompt > 1`` — reuse shared prompt KV.

Memory management
-----------------

When memory is **plentiful**:

- Disable ``--adam_offload``; enable ``--overlap_comm``.
- Use ``--colocate_all_models`` (Hybrid Engine), or at least ``--colocate_critic_reward`` + ``--colocate_actor_ref``.

When hitting **OOM** (priority order):

1. Enable ``--packing_samples`` and ``--gradient_checkpointing``.
2. Reduce ``--micro_train_batch_size`` / ``--micro_rollout_batch_size``.
3. Lower ``--vllm_gpu_memory_utilization`` (e.g., 0.6 → 0.5 → 0.4).
4. Enable ``--adam_offload``; raise ``--zero_stage`` (2 → 3).
5. Disable colocation (remove ``--colocate_*``) and move to distributed mode.

Batch size tuning
-----------------

- **Generation**: maximize ``--micro_rollout_batch_size`` and minimize vLLM TP size (prefer more engines over larger TP).
- **Training**: maximize ``--micro_train_batch_size`` with ``--packing_samples`` enabled.
- **Batch relation**: a common choice is ``train_batch_size = rollout_batch_size * n_samples_per_prompt``.

Long context (>8K tokens)
-------------------------

- Enable RingAttention (``--ring_attn_size``) — see :doc:`sequence_parallelism`.
- Keep ``--packing_samples`` on.
- Increase ``--zero_stage`` (typically 3) and watch memory closely.

For the launch recipes see :doc:`hybrid_engine` and :doc:`agent_training`; for error triage see :doc:`troubleshooting`.

Troubleshooting
===============

This page consolidates the most common issues you may hit when running OpenRLHF.

GPU device index / DeepSpeed init errors
---------------------------------------------

If you see GPU device mapping issues (often in DeepSpeed initialization), try:

.. code-block:: bash

   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1  # NVIDIA
   export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1  # AMD

Then restart Ray and rerun the job.

Out-of-memory (OOM)
-------------------

Common mitigations (rough priority):

- Reduce batch sizes (``--micro_train_batch_size``, ``--micro_rollout_batch_size``).
- Reduce vLLM memory fraction (``--vllm_gpu_memory_utilization``).
- Disable colocation (remove ``--colocate_*`` / ``--colocate_all_models``).
- Enable memory savers (``--adam_offload``, ``--gradient_checkpointing``, higher ``--zero_stage``).

See :doc:`performance` and :doc:`hybrid_engine` for detailed tuning.

vLLM hangs / NCCL issues
------------------------

If vLLM hangs during weight sync or you see NCCL-related issues:

- Try ``--enforce_eager`` (disables CUDA graphs).
- Prefer ``--vllm_sync_backend nccl`` on multi-GPU setups.

See :doc:`hybrid_engine` for more troubleshooting tips.

Ray runtime environment problems
--------------------------------

If workers are missing dependencies, let Ray install them via runtime env:

.. code-block:: bash

   --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'

Debug with py-spy (in-container)
--------------------------------

Use `py-spy <https://github.com/benfred/py-spy>`_ to quickly see what a running OpenRLHF Python process is doing on CPU.

1) Install inside the container:

.. code-block:: bash

   pip install py-spy

2) Find the training PID (common keywords: openrlhf / ray / vllm):

.. code-block:: bash

   ps auxww | rg "openrlhf|ray::|train_ppo_ray|train_sft|train_rm|train_dpo|vllm"

3) Attach and inspect:

.. code-block:: bash

   py-spy top --pid <PID>
   py-spy record --pid <PID> --duration 30 -o profile.svg

If attach fails in Docker, start the container with ptrace enabled:

.. code-block:: bash

   docker run ... --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ...


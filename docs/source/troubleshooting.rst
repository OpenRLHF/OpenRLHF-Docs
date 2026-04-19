Troubleshooting
===============

This page consolidates the most common issues you may hit when running OpenRLHF.

"argparse: unrecognized arguments" after upgrade
------------------------------------------------

OpenRLHF 0.10.2 moved every CLI flag under a dotted section prefix. Old flat names
(``--pretrain``, ``--zero_stage``, ``--vllm_num_engines``, ``--learning_rate``, ...) no longer
parse and argparse will error out. Port your launch scripts to the new surface —
:ref:`flag_migration` in :doc:`common_options` has the full old → new table, and every file
under ``examples/scripts/`` has already been migrated.

GPU device index / DeepSpeed init errors
----------------------------------------

If you see GPU device mapping issues (often in DeepSpeed initialization), try:

.. code-block:: bash

   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1  # NVIDIA
   export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1  # AMD

Then restart Ray and rerun the job.

Out-of-memory (OOM)
-------------------

Common mitigations (rough priority):

- Reduce batch sizes (``--train.micro_batch_size``, ``--rollout.micro_batch_size``).
- Reduce vLLM memory fraction (``--vllm.gpu_memory_utilization``).
- Disable colocation (remove ``--train.colocate_*``).
- Enable memory savers (``--ds.adam_offload``, ``--actor.gradient_checkpointing_enable`` /
  ``--model.gradient_checkpointing_enable``, higher ``--ds.zero_stage``).

See :doc:`performance` and :doc:`hybrid_engine` for detailed tuning.

Muon + DeepSpeed compatibility
------------------------------

``--optim muon`` / ``--actor.optim muon`` is **incompatible** with ``--ds.adam_offload`` — DS's
Muon implementation keeps optimizer state on GPU. If you need adam-offload for memory, switch
back to Adam.

``--muon.ns_steps`` and ``--muon.nesterov`` / ``--muon.no_nesterov`` are **placeholders** on
DeepSpeed 0.18.x: the DS ``muon_update()`` kernel hard-codes ``ns_steps=5`` and Nesterov
``True``. Changing them fires a runtime warning and has no effect. These slots are retained for
forward-compat with future DeepSpeed releases.

Muon requires **DeepSpeed ≥ 0.18.2**. On older DS you will see an init-time error when the
``MuonWithAuxAdam`` type is not registered — upgrade DeepSpeed or revert to ``--optim adam``.

vLLM hangs / NCCL issues
------------------------

If vLLM hangs during weight sync or you see NCCL-related issues:

- Try ``--vllm.enforce_eager`` (disables CUDA graphs).
- Prefer ``--vllm.sync_backend nccl`` on multi-GPU setups.

See :doc:`hybrid_engine` for more troubleshooting tips.

Ray runtime environment problems
--------------------------------

If workers are missing dependencies, let Ray install them via runtime env:

.. code-block:: bash

   --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'

Debug with py-spy (in-container)
--------------------------------

Use `py-spy <https://github.com/benfred/py-spy>`_ to quickly see what a running OpenRLHF Python
process is doing on CPU.

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

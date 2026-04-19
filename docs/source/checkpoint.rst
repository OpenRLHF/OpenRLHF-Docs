Checkpointing
=============

Training large models is expensive and long-running, so resumability is essential. OpenRLHF saves
four kinds of state at each checkpoint:

1. **Model weights** — DeepSpeed-format (sharded across ZeRO ranks).
2. **Optimizer + scheduler state** — for exact resume.
3. **Dataset progress** — via a re-implemented `resumable DistributedSampler
   <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_sampler.py>`_,
   so you don't re-train on already-seen data.
4. **(Optional) HuggingFace-format model** — when ``--ckpt.save_hf`` is set, also writes a
   deployment-ready HF checkpoint.

Resuming with ``--ckpt.load_enable`` gracefully falls back to training-from-scratch if the
checkpoint directory exists but contains no valid checkpoint — useful for first-run /
restart-on-failure scripts.

.. contents::
   :local:
   :depth: 2

Core flags
----------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--ckpt.save_steps``
     - Global training steps between checkpoints (``-1`` = never). For PPO, *steps* are
       model-update steps (not mini-batches).
   * - ``--ckpt.path``
     - Directory where checkpoints are written.
   * - ``--ckpt.load_enable``
     - Resume from ``--ckpt.path``. Falls back to training-from-scratch if the directory contains
       no valid checkpoint.
   * - ``--ckpt.save_hf``
     - Also export a HuggingFace-format model at each checkpoint (so you can deploy without
       DeepSpeed).
   * - ``--ckpt.disable_ds``
     - Skip DeepSpeed checkpoints to save disk — **training progress is no longer recoverable**
       (only HF-format models are kept).
   * - ``--ckpt.max_num``
     - Cap on the number of retained checkpoints (oldest are deleted).
   * - ``--ckpt.max_mem``
     - Cap on total checkpoint size in GB.
   * - ``--ds.use_universal_ckpt``
     - Use DeepSpeed Universal Checkpoint format (ZeRO-stage / world-size agnostic).
   * - ``--ckpt.output_dir``
     - Final HuggingFace-format model save path (always written at the end of training).

PPO-only flags
--------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--ckpt.best_metric_key``
     - Eval metric key for **best-checkpoint** saving (e.g., ``eval_default_pass1``). Empty
       string auto-detects the first ``pass1`` metric from your eval set; ``none`` disables
       best-checkpoint saving.
   * - ``--train.enable_ema``
     - Maintain an Exponential Moving Average copy of the policy weights, saved alongside the
       regular model.
   * - ``--train.ema_beta``
     - EMA decay rate (default ``0.992``). Higher = slower averaging.
   * - ``--critic.save_value_network``
     - Also save the critic / value network checkpoint.

Best-checkpoint tracking requires evaluation: set ``--eval.dataset`` and ``--eval.steps`` so the
trainer has a metric to compare. The best checkpoint is written under a separate path so latest-
and best-checkpoint paths don't collide.

DeepSpeed → Universal conversion
--------------------------------

If you change ZeRO stage or world size between runs, convert the DeepSpeed checkpoint to
Universal format first:

.. code-block:: bash

   bash examples/scripts/ckpt_ds_zero_to_universal.sh

then resume with ``--ds.use_universal_ckpt``.

Example: SFT
------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --model.model_name_or_path meta-llama/Meta-Llama-3-8B \
      --data.dataset Open-Orca/OpenOrca \
      --data.input_key question \
      --data.output_key response \
      --data.input_template $'User: {}\nAssistant: ' \
      --data.max_samples 500000 \
      --data.max_len 2048 \
      --ds.packing_samples \
      --train.batch_size 256 \
      --train.micro_batch_size 2 \
      --train.max_epochs 1 \
      --adam.lr 5e-6 \
      --ds.zero_stage 2 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ckpt.output_dir ./checkpoint/llama3-8b-sft \
      --ckpt.path ./ckpt \
      --ckpt.save_steps 200 \
      --ckpt.save_hf \
      --ckpt.load_enable \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --logger.wandb.key {wandb_token}

Example: RL (Ray + vLLM)
------------------------

To enable checkpointing for an RL run, add the four flags below to the launch command in
:doc:`hybrid_engine` (or the distributed version in :doc:`multi-node`):

.. code-block:: bash

   ... \
   --ckpt.save_steps 50 \
   --ckpt.path /openrlhf/examples/checkpoint/ckpt/ \
   --ckpt.save_hf \
   --ckpt.load_enable

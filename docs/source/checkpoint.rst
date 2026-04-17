Checkpointing
=============

Training large models is expensive and long-running, so resumability is essential. OpenRLHF saves four kinds of state at each checkpoint:

1. **Model weights** — DeepSpeed-format (sharded across ZeRO ranks).
2. **Optimizer + scheduler state** — for exact resume.
3. **Dataset progress** — via a re-implemented `resumable DistributedSampler <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_sampler.py>`_, so you don't re-train on already-seen data.
4. **(Optional) HuggingFace-format model** — when ``--save_hf_ckpt`` is set, also writes a deployment-ready HF checkpoint.

Resuming with ``--load_checkpoint`` gracefully falls back to training-from-scratch if the checkpoint directory exists but contains no valid checkpoint — useful for first-run / restart-on-failure scripts.

.. contents::
   :local:
   :depth: 2

Core flags
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--save_steps``
     - Global training steps between checkpoints (``-1`` = never). For PPO, *steps* are model-update steps (not mini-batches).
   * - ``--ckpt_path``
     - Directory where checkpoints are written.
   * - ``--load_checkpoint``
     - Resume from ``--ckpt_path``. Falls back to training-from-scratch if the directory contains no valid checkpoint.
   * - ``--save_hf_ckpt``
     - Also export a HuggingFace-format model at each checkpoint (so you can deploy without DeepSpeed).
   * - ``--disable_ds_ckpt``
     - Skip DeepSpeed checkpoints to save disk — **training progress is no longer recoverable** (only HF-format models are kept).
   * - ``--max_ckpt_num``
     - Cap on the number of retained checkpoints (oldest are deleted).
   * - ``--max_ckpt_mem``
     - Cap on total checkpoint size in GB.
   * - ``--use_ds_universal_ckpt``
     - Use DeepSpeed Universal Checkpoint format (ZeRO-stage / world-size agnostic).
   * - ``--save_path``
     - Final HuggingFace-format model save path (always written at the end of training).

PPO-only flags
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--best_metric_key``
     - Eval metric key for **best-checkpoint** saving (e.g., ``eval_default_pass1``). Empty string auto-detects the first ``pass1`` metric from your eval set; ``none`` disables best-checkpoint saving.
   * - ``--enable_ema``
     - Maintain an Exponential Moving Average copy of the policy weights, saved alongside the regular model.
   * - ``--ema_beta``
     - EMA decay rate (default ``0.992``). Higher = slower averaging.
   * - ``--save_value_network``
     - Also save the critic / value network checkpoint.

Best-checkpoint tracking requires evaluation: set ``--eval_dataset`` and ``--eval_steps`` so the trainer has a metric to compare. The best checkpoint is written under a separate path so latest- and best-checkpoint paths don't collide.

DeepSpeed → Universal conversion
--------------------------------

If you change ZeRO stage or world size between runs, convert the DeepSpeed checkpoint to Universal format first:

.. code-block:: bash

   bash examples/scripts/ckpt_ds_zero_to_universal.sh

then resume with ``--use_ds_universal_ckpt``.

Example: SFT
------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\nAssistant: ' \
      --max_samples 500000 \
      --max_len 2048 \
      --train_batch_size 256 \
      --micro_train_batch_size 2 \
      --max_epochs 1 \
      --learning_rate 5e-6 \
      --zero_stage 2 \
      --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --save_path ./checkpoint/llama3-8b-sft \
      --ckpt_path ./ckpt \
      --save_steps 200 \
      --save_hf_ckpt \
      --load_checkpoint \
      --logging_steps 1 \
      --eval_steps -1 \
      --use_wandb {wandb_token}

Example: RL (Ray + vLLM)
------------------------

To enable checkpointing for an RL run, add the four flags below to the launch command in :doc:`hybrid_engine` (or the distributed version in :doc:`multi-node`):

.. code-block:: bash

   ... \
   --save_steps 50 \
   --ckpt_path /openrlhf/examples/checkpoint/ckpt/ \
   --save_hf_ckpt \
   --load_checkpoint

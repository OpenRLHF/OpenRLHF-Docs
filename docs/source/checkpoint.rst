Checkpointing
=============

Training large models is expensive, so being able to resume on crash is essential. OpenRLHF re-implements a resumable `DistributedSampler <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_sampler.py>`_ and uses DeepSpeed's checkpoint API to save model / optimizer / scheduler state alongside dataset progress.

Flags
-----

- ``--save_steps``: global training steps between checkpoints. For PPO, steps refer to model updates (not mini-batches).
- ``--ckpt_path``: directory where checkpoints are written.
- ``--load_checkpoint``: resume from ``--ckpt_path``. Gracefully falls back to training-from-scratch if the directory exists but contains no valid checkpoint.
- ``--save_hf_ckpt``: also export a HuggingFace-format model at each checkpoint.
- ``--disable_ds_ckpt``: skip DeepSpeed checkpoints to save disk — **training progress is no longer recoverable**.
- ``--max_ckpt_num``: cap the number of retained checkpoints.
- ``--max_ckpt_mem``: cap the total size (GB) of retained checkpoints.
- ``--use_ds_universal_ckpt``: use DeepSpeed universal checkpoint.
- ``--best_metric_key`` *(PPO only)*: eval metric for best-checkpoint saving (e.g., ``eval_default_pass1``). Empty string auto-detects the first ``pass1`` metric; ``none`` disables best-checkpoint saving.
- ``--enable_ema`` / ``--ema_beta`` *(PPO only)*: keep an EMA copy of the policy (default beta ``0.992``).

Example (SFT)
-------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question --output_key response \
      --input_template $'User: {}\nAssistant: ' \
      --train_batch_size 256 --micro_train_batch_size 2 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --zero_stage 2 --max_epochs 1 --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --learning_rate 5e-6 --gradient_checkpointing \
      --save_steps 200 --ckpt_path ./ckpt --save_hf_ckpt --load_checkpoint \
      --logging_steps 1 --eval_steps -1 \
      --use_wandb {wandb_token}

For an RL checkpoint recipe, add ``--save_steps`` / ``--ckpt_path`` / ``--save_hf_ckpt`` / ``--load_checkpoint`` to the PPO launch command in :doc:`hybrid_engine` (or the distributed version in :doc:`multi-node`).

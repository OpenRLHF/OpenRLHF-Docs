Non-RL Methods
==============

This page covers supervised / preference-based / non-RL training methods (e.g., SFT, DPO) available in OpenRLHF.

.. note::
   Earlier versions of OpenRLHF shipped with KTO, PRM, Knowledge Distillation, and ``batch_inference``-based iterative workflows (rejection sampling, iterative DPO, conditional SFT). These modules have been removed from the upstream codebase. If you need them, pin an older release of OpenRLHF.

.. _train_sft:

Supervised Fine-tuning (SFT)
----------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\\nAssistant: ' \
      --train_batch_size 256 \
      --micro_train_batch_size 8 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --zero_stage 2 \
      --max_epochs 1 \
      --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --packing_samples \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options
~~~~~~~

- ``--input_key``: JSON dataset key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--pretrain_mode``: Continue pretrain mode
- ``--packing_samples``: Packing SFT samples
- ``--multiturn``: Enable multi turn fine-tuning loss

.. note::
   OpenRLHF SFT/DPO/PPO/RM trainers support ``--packing_samples`` `using flash_attention <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_

Direct Preference Optimization (DPO)
------------------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_dpo \
      --save_path ./checkpoint/llama3-8b-dpo \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 5e-7 \
      --beta 0.1 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template \
      --chosen_key chosen \
      --rejected_key rejected \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}


Options
~~~~~~~
- ``--chosen_key`` JSON dataset key for chosen conversions
- ``--rejected_key`` JSON dataset key for rejected conversions
- ``--ref_offload`` Offload Reference Model to CPU
- ``--beta`` The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
- ``--ipo`` for IPO loss.
- ``--label_smoothing`` for cDPO loss.
- ``--packing_samples``: Packing DPO samples
- ``--nll_loss_coef``: Regularization with NLL loss (See Llama 3.1 tech report)

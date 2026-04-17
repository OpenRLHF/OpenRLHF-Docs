Supervised & Preference Training (SFT / RM / DPO)
==================================================

This page covers the **non-RL** trainers in OpenRLHF: supervised fine-tuning, reward-model training, and direct preference optimization. They are the standard preludes to RL training (SFT → RM → PPO) but are also useful on their own.

For RL training (PPO, REINFORCE++, GRPO, RLOO, custom rewards, multi-turn agents, VLM) see :doc:`agent_training`. Shared CLI flags are documented in :doc:`common_options`.

.. note::
   Earlier versions of OpenRLHF shipped with KTO, PRM, Knowledge Distillation, and ``batch_inference``-based iterative workflows (rejection sampling, iterative DPO, conditional SFT). These modules have been removed from the upstream codebase. If you need them, pin an older release of OpenRLHF.

.. contents::
   :local:
   :depth: 2

.. _train_sft:

Supervised Fine-tuning (SFT)
----------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question --output_key response \
      --input_template $'User: {}\\nAssistant: ' \
      --train_batch_size 256 --micro_train_batch_size 8 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --zero_stage 2 --max_epochs 1 --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --packing_samples \
      --learning_rate 5e-6 --gradient_checkpointing \
      --use_wandb {wandb_token}

SFT-specific flags:

- ``--input_key`` / ``--output_key``: dataset JSON keys for the prompt and the target response.
- ``--apply_chat_template``: render conversations with the tokenizer's chat template (use with ``--input_key`` for chat datasets).
- ``--tokenizer_chat_template``: override the tokenizer's default chat template.
- ``--input_template``: custom Python format string when not using a chat template (e.g., ``$'User: {}\nAssistant: '``).
- ``--multiturn``: train on a *compacted* multi-turn dataset format (loss applies to all assistant turns). **Requires** ``--apply_chat_template``.
- ``--pretrain_mode``: switch to a pre-training-style loss (next-token prediction over the whole sequence) — useful for continued pre-training.
- ``--packing_samples``: pack multiple samples per sequence for ~2-3× speedup with FlashAttention.
- ``--ring_attn_size`` / ``--ring_head_stride``: enable RingAttention for long contexts (see :doc:`sequence_parallelism`).
- ``--aux_loss_coef``: MoE balancing loss coefficient (set ``> 0`` for MoE models).

.. note::
   OpenRLHF SFT/DPO/PPO/RM trainers all support ``--packing_samples`` (`packing reference <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_).

.. _train_rm:

Reward Model (RM) Training
--------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --train_batch_size 256 --micro_train_batch_size 4 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 --max_epochs 1 --max_len 8192 \
      --zero_stage 3 --learning_rate 9e-6 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template --chosen_key chosen --rejected_key rejected \
      --attn_implementation flash_attention_2 \
      --packing_samples --gradient_checkpointing \
      --use_wandb {wandb_token}

RM-specific flags:

- ``--chosen_key`` / ``--rejected_key``: dataset JSON keys for the preferred and rejected responses.
- ``--value_head_prefix``: name prefix for the score head (default ``score``). Setting it to ``score`` lets you load the trained model with ``AutoModelForSequenceClassification`` later.
- ``--loss``: RM loss type (default ``sigmoid``).
- ``--margin_loss``: use a margin-based loss (per-sample margin) instead of plain Bradley–Terry; requires a ``margin`` field in the dataset.
- ``--compute_fp32_loss``: compute the RM loss in FP32 for numerical stability (useful when bf16 loss is unstable).
- ``--packing_samples``: pack RM samples for speedup.

Loading the trained RM with ``--value_head_prefix score``:

.. code-block:: python

   reward_model = AutoModelForSequenceClassification.from_pretrained(
       reward_model_path,
       num_labels=1,
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
       use_cache=False,
   )
   # inputs: left-padded token IDs
   reward = reward_model.model(*inputs).last_hidden_state
   reward = reward_model.score(reward)[:, -1]

Direct Preference Optimization (DPO)
------------------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_dpo \
      --save_path ./checkpoint/llama3-8b-dpo \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --train_batch_size 256 --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 --max_epochs 1 --max_len 8192 \
      --zero_stage 3 --learning_rate 5e-7 --beta 0.1 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template --chosen_key chosen --rejected_key rejected \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

DPO-specific flags:

- ``--chosen_key`` / ``--rejected_key``: dataset JSON keys for the preference pair.
- ``--beta``: DPO temperature. Higher β → stay closer to the reference policy; lower β → diverge more aggressively. Typical range: 0.1–0.5.
- ``--ipo``: switch to the IPO loss (`Identity Preference Optimization <https://arxiv.org/abs/2310.12036>`_).
- ``--label_smoothing``: enable cDPO with label smoothing in ``(0, 0.5)``.
- ``--nll_loss_coef``: add an NLL regularization term on the chosen response (per the Llama 3.1 tech report) — often improves stability.
- ``--ref_offload``: offload the reference model to CPU between forward passes to save GPU memory.
- ``--min_lr_ratio``: minimum learning rate as a fraction of the initial LR (used by ``cosine_with_min_lr`` scheduler; default ``0`` means decays to 0).
- ``--packing_samples``: pack DPO samples.

LoRA / QLoRA
------------

All three trainers above support LoRA and QLoRA — set ``--lora_rank > 0`` (and ``--load_in_4bit`` for QLoRA). When LoRA is enabled, only the adapter weights are saved; merge them back with the base model using ``openrlhf.cli.lora_combiner`` (see :doc:`common_options`).

.. note::
   Ray + vLLM (PPO) does **not** currently support LoRA — LoRA is only available for SFT / RM / DPO.

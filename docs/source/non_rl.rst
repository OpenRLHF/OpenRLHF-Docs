Supervised & Preference Training (SFT / RM / DPO)
==================================================

This page covers the **non-RL** trainers in OpenRLHF: supervised fine-tuning, reward-model
training, and direct preference optimization. They are the standard preludes to RL training
(SFT → RM → PPO) but are also useful on their own.

For RL training (PPO, REINFORCE++, GRPO, RLOO, custom rewards, multi-turn agents, VLM) see
:doc:`agent_training`. Shared CLI flags are documented in :doc:`common_options`.

.. note::
   All three trainers accept the **0.10.2 hierarchical CLI**. Single-model config lives under
   ``--model.*``; optimizer under ``--adam.*`` / ``--muon.*`` (selected by ``--optim``);
   scheduler and gradient clip stay flat (``--lr_scheduler``, ``--lr_warmup_ratio``,
   ``--min_lr_ratio``, ``--max_norm``). Old flat flags like ``--pretrain`` or ``--learning_rate``
   no longer parse — see :ref:`flag_migration`.

.. note::
   Earlier versions of OpenRLHF shipped with KTO, PRM, Knowledge Distillation, and
   ``batch_inference``-based iterative workflows (rejection sampling, iterative DPO, conditional
   SFT). These modules have been removed from the upstream codebase. If you need them, pin an
   older release of OpenRLHF.

.. contents::
   :local:
   :depth: 2

.. _train_sft:

Supervised Fine-tuning (SFT)
----------------------------

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
      --train.micro_batch_size 8 \
      --train.max_epochs 1 \
      --adam.lr 5e-6 \
      --ds.zero_stage 2 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ckpt.output_dir ./checkpoint/llama3-8b-sft \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --logger.wandb.key {wandb_token}

SFT-specific flags:

- ``--data.input_key`` / ``--data.output_key``: dataset JSON keys for the prompt and the target
  response.
- ``--data.apply_chat_template``: render conversations with the tokenizer's chat template
  (use with ``--data.input_key`` for chat datasets).
- ``--data.tokenizer_chat_template``: override the tokenizer's default chat template.
- ``--data.input_template``: custom Python format string when not using a chat template (e.g.,
  ``$'User: {}\nAssistant: '``).
- ``--data.multiturn``: train on a *compacted* multi-turn dataset format (loss applies to all
  assistant turns). **Requires** ``--data.apply_chat_template``.
- ``--model.pretrain_mode_enable``: switch to a pre-training-style loss (next-token prediction
  over the whole sequence) — useful for continued pre-training.
- ``--ds.packing_samples``: pack multiple samples per sequence for ~2–3× speedup with
  FlashAttention.
- ``--ds.ring_attn_size`` / ``--ds.ring_attn_head_stride``: enable RingAttention for long
  contexts (see :doc:`sequence_parallelism`).
- ``--model.aux_loss_coef``: MoE balancing loss coefficient (set ``> 0`` for MoE models).

.. note::
   OpenRLHF SFT / DPO / PPO / RM trainers all support ``--ds.packing_samples``
   (`packing reference <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_).

.. _train_rm:

Reward Model (RM) Training
--------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --model.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
      --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --data.apply_chat_template \
      --data.chosen_key chosen \
      --data.rejected_key rejected \
      --data.max_len 8192 \
      --ds.packing_samples \
      --train.batch_size 256 \
      --train.micro_batch_size 4 \
      --train.max_epochs 1 \
      --adam.lr 9e-6 \
      --ds.zero_stage 3 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ckpt.output_dir ./checkpoint/llama3-8b-rm \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --logger.wandb.key {wandb_token}

RM-specific flags:

- ``--data.chosen_key`` / ``--data.rejected_key``: dataset JSON keys for the preferred and
  rejected responses.
- ``--ds.value_head_prefix``: name prefix for the score head (default ``score``). Setting it
  to ``score`` lets you load the trained model with ``AutoModelForSequenceClassification`` later.
- ``--model.loss_type``: RM loss type (default ``sigmoid``).
- ``--model.margin_loss_enable``: use a margin-based loss (per-sample margin) instead of plain
  Bradley–Terry; requires a ``margin`` field in the dataset.
- ``--model.compute_fp32_loss_enable``: compute the RM loss in FP32 for numerical stability
  (useful when bf16 loss is unstable).
- ``--ds.packing_samples``: pack RM samples for speedup.

Loading the trained RM with ``--ds.value_head_prefix score``:

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
      --model.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
      --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --data.apply_chat_template \
      --data.chosen_key chosen \
      --data.rejected_key rejected \
      --data.max_len 8192 \
      --ds.packing_samples \
      --train.batch_size 256 \
      --train.micro_batch_size 1 \
      --train.max_epochs 1 \
      --adam.lr 5e-7 \
      --model.beta 0.1 \
      --ds.zero_stage 3 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ckpt.output_dir ./checkpoint/llama3-8b-dpo \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --logger.wandb.key {wandb_token}

DPO-specific flags:

- ``--data.chosen_key`` / ``--data.rejected_key``: dataset JSON keys for the preference pair.
- ``--model.beta``: DPO temperature. Higher β → stay closer to the reference policy; lower β →
  diverge more aggressively. Typical range: 0.1–0.5.
- ``--model.ipo_enable``: switch to the IPO loss (`Identity Preference Optimization
  <https://arxiv.org/abs/2310.12036>`_).
- ``--model.label_smoothing``: enable cDPO with label smoothing in ``(0, 0.5)``.
- ``--model.nll_loss_coef``: add an NLL regularization term on the chosen response (per the
  Llama 3.1 tech report) — often improves stability.
- ``--ref.offload``: offload the reference model to CPU between forward passes to save GPU
  memory.
- ``--min_lr_ratio``: minimum learning rate as a fraction of the initial LR (used by
  ``cosine_with_min_lr`` scheduler; default ``0.1``).
- ``--ds.packing_samples``: pack DPO samples.

Optimizer: Adam or Muon
-----------------------

All three trainers expose the **same optimizer switch**. Pick the optimizer with ``--optim`` and
configure it via the matching section (leave the other section at defaults — it's ignored):

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Switch
     - Active section
     - Ignored section
   * - ``--optim adam`` *(default)*
     - ``--adam.lr`` / ``--adam.betas`` / ``--adam.eps`` / ``--adam.weight_decay``
     - ``--muon.*``
   * - ``--optim muon``
     - ``--muon.lr`` / ``--muon.momentum`` + ``--adam.lr`` / ``--adam.betas`` / ``--adam.eps`` /
       ``--adam.weight_decay`` (the aux-Adam subgroup for embeddings / LM head / 1-D params
       reuses every ``--adam.*`` knob; the LR scheduler drives both groups)
     - none — ``--adam.*`` is always active under Muon

**Muon example (SFT)** — actor-side DS-Muon for a Llama-3-8B run:

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --model.model_name_or_path meta-llama/Meta-Llama-3-8B \
      --data.dataset Open-Orca/OpenOrca \
      --data.input_key question --data.output_key response \
      --data.max_len 2048 --ds.packing_samples \
      --train.batch_size 256 --train.micro_batch_size 8 \
      --train.max_epochs 1 \
      --optim muon \
      --muon.lr 0.02 \
      --muon.momentum 0.95 \
      --adam.lr 5e-6 \
      --adam.weight_decay 0.0 \
      --lr_scheduler cosine_with_min_lr \
      --lr_warmup_ratio 0.03 \
      --min_lr_ratio 0.1 \
      --max_norm 1.0 \
      --ds.zero_stage 2 --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ckpt.output_dir ./checkpoint/llama3-8b-sft-muon

.. note::
   **Muon requirements and caveats**

   - DeepSpeed **≥ 0.18.2** is required; OpenRLHF builds the DS Muon config on your behalf.
   - ``--ds.adam_offload`` is **incompatible** with Muon — DS's Muon keeps optimizer state on
     GPU. Keep ``--ds.adam_offload`` disabled when using ``--optim muon``.
   - ``--muon.ns_steps`` and ``--muon.nesterov`` are **placeholders**: DeepSpeed 0.18.x
     hard-codes ``ns_steps=5`` and Nesterov ``True`` inside ``muon_update()``. Changing them
     produces a runtime warning and no behavior change. Exposed now for forward-compat with
     future DS releases.
   - Muon only updates 2-D hidden weight matrices. Embeddings, the LM head, and 1-D parameters
     (LayerNorm / biases) are optimized by the aux-Adam subgroup at ``--adam.lr``.
   - **Weight decay is shared.** DeepSpeed 0.18.x stamps a single ``weight_decay`` into both
     groups; OpenRLHF reads it from ``--adam.weight_decay``. There is no separate
     ``--muon.weight_decay`` flag.
   - **No bias / LayerNorm exemption under Muon.** The pure-Adam path exempts bias and
     LayerNorm from weight decay; under Muon, DS refuses param-group dicts and applies the
     same ``weight_decay`` to every parameter.

LoRA / QLoRA
------------

All three trainers above support LoRA and QLoRA — set ``--ds.lora.rank > 0`` (and
``--ds.load_in_4bit`` for QLoRA). When LoRA is enabled, only the adapter weights are saved;
merge them back with the base model using ``openrlhf.cli.lora_combiner`` (see :doc:`common_options`).

Example — SFT + LoRA on Mixtral:

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --model.model_name_or_path mistralai/Mixtral-8x7B-v0.1 \
      --data.dataset Open-Orca/OpenOrca \
      --data.max_len 2048 --ds.packing_samples \
      --train.batch_size 128 --train.micro_batch_size 1 \
      --train.max_epochs 1 \
      --adam.lr 5e-6 \
      --ds.zero_stage 3 --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ds.lora.rank 64 --ds.lora.alpha 64 \
      --ckpt.output_dir ./checkpoint/mixtral-sft-lora

.. note::
   Ray + vLLM (PPO) does **not** currently support LoRA — LoRA is only available for SFT / RM /
   DPO.

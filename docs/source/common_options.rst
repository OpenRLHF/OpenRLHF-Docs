Common CLI Options
==================

OpenRLHF 0.10.2 organizes every CLI flag under a **dotted section prefix** that mirrors
ownership: ``--ds.*`` for DeepSpeed, ``--vllm.*`` for vLLM, ``--rollout.*`` for generation
knobs, ``--data.*`` / ``--train.*`` / ``--eval.*`` / ``--ckpt.*`` / ``--logger.*`` / ``--algo.*``
for pipeline stages, and per-entity ``--actor.*`` / ``--critic.*`` / ``--ref.*`` / ``--reward.*``
for PPO or ``--model.*`` for the single-model trainers (SFT / RM / DPO). Flat aliases from
earlier releases were removed — every launch script must use the new names.

This page is the central flag reference. Trainer-specific knobs are on :doc:`non_rl` and
:doc:`agent_training`; end-to-end launch scripts live under `examples/scripts
<https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

.. contents::
   :local:
   :depth: 2

Section map
-----------

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Prefix
     - Who owns it
     - What lives under it
   * - ``--ds.*``
     - DeepSpeed + model loading
     - ``zero_stage``, ``param_dtype``, ``adam_offload``, ``tensor_parallel_size``,
       ``zpg``, ``overlap_comm``, ``grad_accum_dtype``, ``deepcompile``, ``enable_sleep``,
       ``ring_attn_size``, ``ring_attn_head_stride``, ``use_universal_ckpt``,
       ``attn_implementation``, ``experts_implementation``, ``use_liger_kernel``,
       ``load_in_4bit``, ``lora.{rank, alpha, dropout, target_modules}``,
       ``packing_samples``, ``value_head_prefix`` (RM / PPO reward head).
   * - ``--vllm.*``
     - vLLM generation engine
     - ``num_engines``, ``tensor_parallel_size``, ``gpu_memory_utilization``, ``sync_backend``,
       ``sync_with_ray``, ``enforce_eager``, ``enable_prefix_caching``, ``enable_sleep``.
   * - ``--rollout.*``
     - Rollout / sampling
     - ``batch_size``, ``micro_batch_size``, ``n_samples_per_prompt``, ``temperature``,
       ``top_p``, ``max_new_tokens``, ``max_tokens_per_gpu``, ``vllm_generate_batch_size``.
   * - ``--data.*``
     - Dataset + tokenization
     - ``prompt_dataset`` (PPO) or ``dataset`` (SFT/RM/DPO), split / probs, key mapping,
       templating, ``max_len``, ``max_samples``, ``image_key``,
       ``max_images_per_prompt``, ``multiturn``, ``dataloader_num_workers``.
   * - ``--train.*``
     - Training orchestration
     - ``batch_size``, ``micro_batch_size``, ``max_tokens_per_gpu``, ``max_epochs``,
       ``num_episodes``, ``seed``, ``full_determinism_enable``, ``async_enable`` /
       ``async_queue_size`` / ``partial_rollout_enable``, ``dynamic_batch_enable``,
       ``enable_ema`` / ``ema_beta``, ``agent_func_path``, ``colocate_actor_ref`` /
       ``colocate_critic_reward`` / ``colocate_all``.
   * - ``--eval.*``
     - Evaluation
     - ``dataset``, ``split``, ``steps``, ``temperature``, ``n_samples_per_prompt``.
   * - ``--ckpt.*``
     - Checkpointing
     - ``output_dir``, ``path``, ``save_steps``, ``save_hf``, ``disable_ds``, ``max_num``,
       ``max_mem``, ``load_enable``, ``best_metric_key``.
   * - ``--logger.*``
     - Logging
     - ``logging_steps``, ``tensorboard_dir``, ``wandb.{key, org, group, project, run_name}``.
   * - ``--algo.*``
     - RL algorithm (PPO only)
     - ``advantage.{estimator, gamma, lambd, no_std_norm, is_correction_enable,
       is_correction_type, is_correction_threshold}``,
       ``kl.{init_coef, target, horizon, estimator, use_loss}``,
       ``dynamic_filtering_enable``, ``dynamic_filtering_range``.
   * - ``--actor.*`` / ``--critic.*`` / ``--ref.*`` / ``--reward.*``
     - Per-role model config (PPO)
     - ``model_name_or_path``, ``num_nodes``, ``num_gpus_per_node``,
       ``gradient_checkpointing_enable``, ``freeze_visual_encoder`` (actor),
       ``optim`` / ``adam.*`` / ``muon.*``, ``lr_scheduler`` / ``lr_warmup_ratio`` /
       ``min_lr_ratio`` / ``max_norm``, plus role-specific loss / clip / offload knobs.
       Engine-level model loading (``attn_implementation``, ``lora.*``, ``packing_samples``,
       ``load_in_4bit``, ``use_liger_kernel``) lives under ``--ds.*`` and is shared across
       roles.
   * - ``--model.*``
     - Single-model loss/loader config (SFT / RM / DPO)
     - ``model_name_or_path``, ``gradient_checkpointing_enable``, ``aux_loss_coef``,
       ``beta`` (DPO), ``ipo_enable``, ``label_smoothing``, ``nll_loss_coef``,
       ``loss_type`` / ``compute_fp32_loss_enable`` / ``margin_loss_enable`` (RM),
       ``pretrain_mode_enable`` (SFT). Engine-level model loading and LoRA live under
       ``--ds.*``.
   * - (flat)
     - Launcher & special
     - ``--optim``, ``--lr_scheduler``, ``--lr_warmup_ratio``, ``--min_lr_ratio``,
       ``--max_norm`` (flat for SFT/RM/DPO — per-entity in PPO), ``--local_rank``
       (DeepSpeed injects), ``--use_ms`` (ModelScope).

.. note::
   Naming convention: boolean toggles whose names do not already read as predicates carry an
   explicit ``_enable`` suffix, e.g. ``--reward.normalize_enable``, ``--ckpt.load_enable``,
   ``--train.async_enable``, ``--train.dynamic_batch_enable``, ``--algo.dynamic_filtering_enable``,
   ``--algo.advantage.is_correction_enable``. Flags already shaped like verbs
   (``freeze_*``, ``save_*``, ``enforce_*``, ``colocate_*``, ``use_*``, ``*_offload``) stay bare.

DeepSpeed
---------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--ds.zero_stage``
     - DeepSpeed ZeRO stage (0 / 1 / 2 / 3). Use 3 for large models.
   * - ``--ds.tensor_parallel_size``
     - DeepSpeed tensor parallelism (AutoTP) size — only with ``--ds.zero_stage 0/1/2``.
   * - ``--ds.adam_offload``
     - Offload the Adam optimizer state to CPU; saves GPU memory at the cost of step time.
       **Not compatible with Muon** — DS's Muon implementation keeps optimizer state on GPU.
   * - ``--ds.param_dtype``
     - Parameter dtype: ``bf16`` (default) or ``fp16``.
   * - ``--ds.zpg``
     - ZeRO++ max partition size (default ``1``).
   * - ``--ds.overlap_comm``
     - Overlap backward with gradient reduce (larger bucket budget — needs more memory).
   * - ``--ds.grad_accum_dtype``
     - Adam grad-accumulation dtype.
   * - ``--ds.deepcompile``
     - Enable `DeepCompile <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md>`_
       graph compilation (PyTorch 2.0+).
   * - ``--ds.enable_sleep``
     - DeepSpeed sleep mode — free DeepSpeed memory between training steps (paired with
       ``--vllm.enable_sleep`` + ``--train.colocate_all`` for Hybrid Engine).
   * - ``--ds.ring_attn_size`` / ``--ds.ring_attn_head_stride``
     - RingAttention sequence-parallel group / head stride (see :doc:`sequence_parallelism`).
   * - ``--ds.use_universal_ckpt``
     - Use DeepSpeed Universal Checkpoint format (ZeRO-stage / world-size agnostic).
   * - ``--ds.attn_implementation``
     - Attention backend: ``eager`` / ``flash_attention_2`` / ``flash_attention_3`` /
       ``kernels-community/vllm-flash-attn3``. Shared by every model loaded by the trainer
       (actor / critic / reward / reference, or the single SFT/RM/DPO model).
   * - ``--ds.experts_implementation``
     - MoE expert computation strategy passed through to ``transformers.from_pretrained``.
       ``None`` (default) lets transformers auto-pick (``grouped_mm`` when supported, else
       ``eager``); explicit choices are ``eager`` / ``batched_mm`` / ``grouped_mm`` /
       ``deepgemm``.
   * - ``--ds.use_liger_kernel``
     - Enable `Liger Kernel <https://github.com/linkedin/Liger-Kernel>`_ for fused ops.
   * - ``--ds.load_in_4bit``
     - Load the base model in 4-bit (QLoRA — pair with ``--ds.lora.*``).
   * - ``--ds.lora.{rank, alpha, dropout, target_modules}``
     - LoRA / QLoRA config (set ``--ds.lora.rank > 0`` to enable). SFT / RM / DPO only —
       Ray + vLLM PPO does not support LoRA.
   * - ``--ds.packing_samples``
     - Pack multiple samples per sequence (Flash-Attention path). Large throughput win.
   * - ``--ds.value_head_prefix``
     - Score-head name prefix for sequence-regression models (RM training, PPO reward
       model loader, ``serve_rm``). Default ``score``.

vLLM
----

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--vllm.num_engines``
     - Number of vLLM engines (``0`` disables vLLM entirely — useful for non-RL trainers).
   * - ``--vllm.tensor_parallel_size``
     - Tensor parallel size per vLLM engine.
   * - ``--vllm.gpu_memory_utilization``
     - KV-cache fraction. Start at ``0.5`` on 8×A100-80G and raise if stable.
   * - ``--vllm.sync_backend``
     - DeepSpeed → vLLM weight sync backend (``nccl`` recommended on multi-GPU).
   * - ``--vllm.sync_with_ray``
     - Use Ray groups (instead of bare NCCL) for weight sync.
   * - ``--vllm.enforce_eager``
     - Disable CUDA graphs in vLLM (avoids some hang modes; reduces memory).
   * - ``--vllm.enable_prefix_caching``
     - Enable vLLM prefix caching (pairs well with ``--rollout.n_samples_per_prompt > 1``).
   * - ``--vllm.enable_sleep``
     - vLLM sleep mode — free most of vLLM's memory between rollouts. Hybrid Engine only;
       incompatible with ``--train.async_enable``.

Rollout / generation
--------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--rollout.batch_size``
     - Prompt batch size per rollout step (experience-collection batch).
   * - ``--rollout.micro_batch_size``
     - Per-GPU micro-batch during the forward over generated samples (e.g., log-prob recompute).
   * - ``--rollout.n_samples_per_prompt``
     - Completions per prompt. Must be ``> 1`` for RLOO / REINFORCE++-baseline / GRPO / dynamic
       filtering.
   * - ``--rollout.temperature`` / ``--rollout.top_p``
     - vLLM sampling parameters during rollouts.
   * - ``--rollout.max_new_tokens``
     - Max tokens to generate per sample. ``None`` → dynamically ``max_len - prompt_len``.
   * - ``--rollout.max_tokens_per_gpu``
     - Token budget per GPU for the rollout-side forward (used with ``--train.dynamic_batch_enable``).
   * - ``--rollout.vllm_generate_batch_size``
     - vLLM generation batch. If larger than ``--rollout.batch_size``, requires
       ``--train.async_enable`` (oversampling buffers extra batches in the async queue).

Data
----

Shared across trainers. Note that the **prompt-input** flag differs: PPO uses
``--data.prompt_dataset`` (the value is a dataset path — renamed from the old ``--data.prompt``),
while SFT / RM / DPO use ``--data.dataset``.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--data.prompt_dataset`` *(PPO)* / ``--data.dataset`` *(SFT/RM/DPO)*
     - HuggingFace dataset name(s) or path(s). Comma-separate for mixing.
   * - ``--data.prompt_probs`` *(PPO)* / ``--data.dataset_probs`` *(SFT/RM/DPO)*
     - Sampling probabilities when mixing multiple datasets (e.g., ``0.1,0.4,0.5``).
   * - ``--data.prompt_split`` *(PPO)* / ``--data.dataset_split`` *(SFT/RM/DPO)*
     - HF split name (default ``train``).
   * - ``--data.input_key`` / ``--data.output_key`` / ``--data.label_key``
     - JSON keys for prompts, SFT targets, and RL ground-truth labels.
   * - ``--data.prompt_key`` / ``--data.chosen_key`` / ``--data.rejected_key``
     - Preference-dataset keys (RM / DPO).
   * - ``--data.image_key``
     - VLM image-path / URL key (default ``images``).
   * - ``--data.apply_chat_template``
     - Apply the tokenizer's chat template.
   * - ``--data.tokenizer_chat_template``
     - Override the tokenizer's default chat template.
   * - ``--data.input_template``
     - Plain-text format string when not using a chat template (e.g., ``$'User: {}\nAssistant: '``).
   * - ``--data.max_len``
     - Max total sequence length (prompt + response).
   * - ``--data.max_samples``
     - Cap on training samples.
   * - ``--data.dataloader_num_workers``
     - DataLoader workers (default ``0``; for Ray training, ensure enough CPUs per actor).
   * - ``--data.disable_fast_tokenizer``
     - Force ``use_fast=False`` on ``AutoTokenizer``.
   * - ``--data.multiturn``
     - *(SFT)* Train on compacted multi-turn chat data (loss applies to all assistant turns).
       Requires ``--data.apply_chat_template``.
   * - ``--data.max_images_per_prompt``
     - *(PPO VLM)* Max images per prompt for vLLM (``0`` = text-only).

Training
--------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--train.batch_size`` / ``--train.micro_batch_size``
     - Global and per-GPU training batch sizes.
   * - ``--train.max_tokens_per_gpu``
     - Token budget per GPU during training (used with ``--train.dynamic_batch_enable``).
   * - ``--train.max_epochs``
     - PPO optimization epochs per rollout (or epochs over dataset for SFT / RM / DPO).
   * - ``--train.num_episodes`` *(PPO)*
     - Total RL episodes (one episode = full pass through the prompt set).
   * - ``--train.seed``
     - Global random seed (default ``42``).
   * - ``--train.full_determinism_enable``
     - Bit-reproducible behavior (slower; vLLM v1 + fixed seed paths).
   * - ``--train.dynamic_batch_enable``
     - Form micro-batches by token budget instead of count — much better packing for
       variable-length sequences. Pair with ``--train.max_tokens_per_gpu`` /
       ``--rollout.max_tokens_per_gpu``.
   * - ``--train.async_enable`` / ``--train.async_queue_size`` / ``--train.partial_rollout_enable``
     - Async pipeline controls (see :doc:`async_training`).
   * - ``--train.enable_ema`` / ``--train.ema_beta``
     - Track an EMA copy of the actor (see :doc:`checkpoint`).
   * - ``--train.agent_func_path``
     - Multi-turn agent Python file. Sets up the multi-turn executor (see :doc:`agent_training`).
   * - ``--train.colocate_actor_ref`` / ``--train.colocate_critic_reward`` / ``--train.colocate_all``
     - Role colocation for Ray / vLLM placement (see :doc:`hybrid_engine`).

Evaluation
----------

- ``--eval.dataset``: eval dataset path (with ``--data.prompt_dataset`` for PPO; reward-function
  runs only).
- ``--eval.split``: eval split (default differs per trainer).
- ``--eval.steps``: evaluate every *N* training steps (``-1`` to disable).
- ``--eval.temperature`` / ``--eval.n_samples_per_prompt`` *(PPO)*: sampling params for eval
  rollouts.

Checkpointing
-------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--ckpt.output_dir``
     - Final HuggingFace-format model save path (always written at end of training).
   * - ``--ckpt.path``
     - Directory for resumable DeepSpeed-format checkpoints.
   * - ``--ckpt.save_steps``
     - Steps between checkpoints (``-1`` = never).
   * - ``--ckpt.save_hf``
     - Also export an HF-format model at every checkpoint.
   * - ``--ckpt.disable_ds``
     - Skip DeepSpeed checkpoints — training progress is not recoverable.
   * - ``--ckpt.max_num`` / ``--ckpt.max_mem``
     - Cap retained checkpoint count / total size (GB).
   * - ``--ckpt.load_enable``
     - Resume from ``--ckpt.path``; gracefully falls back to fresh training if no valid ckpt.
   * - ``--ckpt.best_metric_key`` *(PPO)*
     - Eval metric key for best-checkpoint saving (empty = auto-detect first ``pass1``;
       ``none`` = disable).

Detailed semantics and SFT/PPO examples live in :doc:`checkpoint`.

Logging
-------

- ``--logger.logging_steps``: log every *N* training steps.
- ``--logger.wandb.key {token-or-True}``: Wandb logging (``True`` uses a prior ``wandb login``).
- ``--logger.wandb.org`` / ``--logger.wandb.group`` / ``--logger.wandb.project`` /
  ``--logger.wandb.run_name``: Wandb metadata.
- ``--logger.tensorboard_dir {logdir}``: TensorBoard logging path.

Model / entity flags
--------------------

**PPO** routes model config through four parallel sections:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Prefix
     - Scope
   * - ``--actor.*``
     - The policy (trained). Owns optimizer / scheduler / grad-clip, loss knobs (``eps_clip``,
       ``dual_clip``, ``entropy_coef``, ``aux_loss_coef``, ``policy_loss_type``,
       ``entropy_coef``), ``gradient_checkpointing_enable``, ``freeze_visual_encoder``.
   * - ``--critic.*``
     - Value network. Owns its own optimizer section, ``value_clip``, ``save_value_network``,
       ``freezing_steps`` (freeze **actor** while critic warms up — despite the name it lives
       under the critic section, paired with critic training).
   * - ``--ref.*``
     - Frozen reference model. ``offload`` moves it to CPU between forwards.
   * - ``--reward.*``
     - Reward model / function. Owns ``model_name_or_path``, ``remote_url`` (HTTP RM or local
       ``.py`` reward function), ``normalize_enable``, ``clip_range``,
       ``overlong_buffer_len`` / ``overlong_penalty_factor``, ``stop_properly_penalty_coef``,
       ``offload``. The score-head prefix lives under ``--ds.value_head_prefix``.

Each PPO entity carries its own ``--{entity}.num_nodes`` / ``--{entity}.num_gpus_per_node``.

Engine-level model loading flags — ``--ds.attn_implementation``,
``--ds.experts_implementation``, ``--ds.use_liger_kernel``, ``--ds.load_in_4bit``,
``--ds.lora.*``, ``--ds.packing_samples`` — are **shared** across every role; you cannot set
a different attention implementation or LoRA config per-entity.

**SFT / RM / DPO** use the single-model namespace ``--model.*``:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--model.model_name_or_path``
     - HF model name or path.
   * - ``--model.gradient_checkpointing_enable`` / ``--model.gradient_checkpointing_reentrant``
     - Trade compute for memory (recompute activations); reentrant toggle.
   * - ``--model.aux_loss_coef``
     - MoE balancing loss coefficient (``> 0`` for MoE models).
   * - ``--model.beta`` *(DPO)*
     - DPO temperature.
   * - ``--model.ipo_enable`` / ``--model.label_smoothing`` / ``--model.nll_loss_coef`` *(DPO)*
     - IPO switch / cDPO smoothing / NLL regularization.
   * - ``--model.loss_type`` / ``--model.compute_fp32_loss_enable`` /
       ``--model.margin_loss_enable`` *(RM)*
     - Reward-model loss family. Score-head prefix is ``--ds.value_head_prefix``.
   * - ``--model.pretrain_mode_enable`` *(SFT)*
     - Pretrain-style loss (next-token over whole sequence).

Engine-level model loading (``attn_implementation``, ``experts_implementation``,
``use_liger_kernel``, ``load_in_4bit``, ``lora.*``, ``packing_samples``) lives under
``--ds.*`` for SFT / RM / DPO too.

Optimizer: Adam or Muon
-----------------------

0.10.2 exposes the optimizer as two parallel sections under ``--adam.*`` and ``--muon.*``,
selected at runtime by a single switch:

- ``--optim {adam, muon}`` in SFT / RM / DPO.
- ``--actor.optim {adam, muon}`` and ``--critic.optim {adam, muon}`` in PPO (actor / critic
  are independent — actor-Muon with critic-Adam is supported out of the box).

Pure Adam
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--adam.lr`` *(SFT/RM/DPO)* / ``--actor.adam.lr`` / ``--critic.adam.lr`` *(PPO)*
     - Learning rate.
   * - ``--adam.betas``
     - Adam β₁/β₂ (default ``(0.9, 0.95)``).
   * - ``--adam.eps``
     - Adam ε (default ``1e-8``).
   * - ``--adam.weight_decay``
     - L2 weight decay (replaces the old flat ``--l2``).

Muon (``--optim muon``)
~~~~~~~~~~~~~~~~~~~~~~~

DeepSpeed's ``MuonWithAuxAdam`` uses Muon for 2-D hidden weight matrices and a parallel AdamW
subgroup for embeddings / LM head / 1-D params. In 0.10.2 OpenRLHF drives this directly:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--muon.lr``
     - LR for the Muon 2-D-weight group (default ``0.02``).
   * - ``--muon.momentum``
     - Muon momentum (default ``0.95``).
   * - ``--muon.ns_steps``
     - Newton–Schulz iteration count. **Placeholder** — DeepSpeed 0.18.x hard-codes
       ``ns_steps=5`` inside ``muon_update()``; a runtime warning fires if you change it.
       Exposed for forward-compat with future DeepSpeed releases.
   * - ``--muon.nesterov`` / ``--muon.no_nesterov``
     - Nesterov momentum toggle. **Placeholder** — DeepSpeed 0.18.x hard-codes Nesterov on.

The aux-Adam subgroup (embeddings / LM head / 1-D / value head) reuses ``--adam.lr``,
``--adam.betas``, ``--adam.eps``, ``--adam.weight_decay`` — and the LR scheduler drives
**both** groups simultaneously. Per-entity in PPO: ``--actor.muon.lr``,
``--critic.muon.lr``, etc.

.. note::
   **Requirements and caveats**

   - Requires **DeepSpeed ≥ 0.18.2**.
   - Incompatible with ``--ds.adam_offload`` — Muon keeps state on GPU.
   - **Weight decay is shared.** DeepSpeed 0.18.x stamps a single ``weight_decay`` into
     both the Muon and aux-Adam param groups; OpenRLHF therefore reads it from
     ``--adam.weight_decay`` (or ``--{entity}.adam.weight_decay``) and applies it to
     both. There is no separate ``--muon.weight_decay`` flag.
   - **Bias / LayerNorm decay exemption is NOT applied under Muon.** DS Muon partitions
     a flat ``Parameter`` list via ``p.use_muon`` and refuses param-group dicts; splitting
     groups post-init would desync ZeRO's bit16/fp32 metadata. The pure-Adam path still
     exempts bias / LayerNorm from weight decay as usual.
   - **0.10.2 fix.** Previously the aux-Adam subgroup silently inherited Muon's LR
     (``0.02``), which nuked pretrained embeddings over the first ~100 steps. 0.10.2
     emits ``muon_lr`` and ``adam_lr`` explicitly to DS so each group follows its own
     initial LR. Make sure to set ``--adam.lr`` (PPO: ``--actor.adam.lr``) when training
     with ``--optim muon``.

Learning-rate scheduler & gradient clip
---------------------------------------

Flags stay flat in the single-model trainers, nested per-entity in PPO.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - SFT / RM / DPO
     - PPO (per entity)
   * - ``--lr_scheduler``
     - ``--actor.lr_scheduler`` / ``--critic.lr_scheduler``
   * - ``--lr_warmup_ratio``
     - ``--actor.lr_warmup_ratio`` / ``--critic.lr_warmup_ratio``
   * - ``--min_lr_ratio``
     - ``--actor.min_lr_ratio`` / ``--critic.min_lr_ratio``
   * - ``--max_norm``
     - ``--actor.max_norm`` / ``--critic.max_norm``

Default scheduler is ``cosine_with_min_lr`` with ``warmup_ratio 0.03`` and ``min_lr_ratio 0.1``.

LoRA / QLoRA
------------

LoRA / QLoRA is supported by SFT / RM / DPO. **Not supported** by Ray + vLLM PPO.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag (SFT/RM/DPO)
     - Meaning
   * - ``--ds.lora.rank``
     - Set ``> 0`` to enable LoRA (rank of the low-rank update).
   * - ``--ds.lora.alpha``
     - LoRA alpha scaling.
   * - ``--ds.lora.dropout``
     - LoRA dropout (passed through to HuggingFace PEFT).
   * - ``--ds.lora.target_modules``
     - PEFT target modules (e.g., ``q_proj k_proj v_proj o_proj``, or ``all-linear``).
   * - ``--ds.load_in_4bit``
     - Load the base model in 4-bit (QLoRA).

Only the adapter is saved. Merge it back with the base model to deploy:

.. code-block:: bash

   python -m openrlhf.cli.lora_combiner \
      --model_path meta-llama/Meta-Llama-3-8B \
      --lora_path ./checkpoint/llama3-8b-rm \
      --output_path ./checkpoint/llama-3-8b-rm-combined \
      --is_rm \
      --param_dtype bf16

Use ``--is_rm`` when merging a reward model (preserves the score head).

Long context & checkpointing (dedicated pages)
----------------------------------------------

- **RingAttention** — ``--ds.ring_attn_size`` / ``--ds.ring_attn_head_stride``; see
  :doc:`sequence_parallelism`.
- **Checkpointing** — ``--ckpt.*`` + ``--train.enable_ema``; see :doc:`checkpoint`.

.. _flag_migration:

Flag migration (0.9.x / early 0.10 → 0.10.2)
---------------------------------------------

Upgrade checklist: replace the old flat flag on the left with the dotted flag on the right.
Every launch script in ``examples/scripts/`` has already been migrated and is a working
reference.

Models
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--pretrain``
     - ``--actor.model_name_or_path`` *(PPO)* / ``--model.model_name_or_path`` *(SFT/RM/DPO)*
   * - ``--reward_pretrain``
     - ``--reward.model_name_or_path``
   * - ``--ref_pretrain``
     - ``--ref.model_name_or_path``
   * - ``--critic_pretrain``
     - ``--critic.model_name_or_path``
   * - ``--remote_rm_url``
     - ``--reward.remote_url``
   * - ``--attn_implementation``
     - ``--ds.attn_implementation`` *(all trainers — moved here so every model in a PPO run
       shares one backend)*
   * - ``--use_liger_kernel``
     - ``--ds.use_liger_kernel``
   * - ``--load_in_4bit``
     - ``--ds.load_in_4bit``
   * - ``--gradient_checkpointing``
     - ``--actor.gradient_checkpointing_enable`` / ``--model.gradient_checkpointing_enable``
   * - ``--gradient_checkpointing_use_reentrant``
     - ``--actor.gradient_checkpointing_reentrant`` / ``--model.gradient_checkpointing_reentrant``
   * - ``--freeze_visual_encoder``
     - ``--actor.freeze_visual_encoder``
   * - ``--lora_rank`` / ``--lora_alpha`` / ``--lora_dropout`` / ``--target_modules``
     - ``--ds.lora.rank`` / ``...alpha`` / ``...dropout`` / ``...target_modules``

Ray placement / colocation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--{actor,critic,ref,reward}_num_nodes``
     - ``--{..}.num_nodes``
   * - ``--{actor,critic,ref,reward}_num_gpus_per_node``
     - ``--{..}.num_gpus_per_node``
   * - ``--colocate_actor_ref``
     - ``--train.colocate_actor_ref``
   * - ``--colocate_critic_reward``
     - ``--train.colocate_critic_reward``
   * - ``--colocate_all_models``
     - ``--train.colocate_all``

vLLM / DeepSpeed
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--vllm_num_engines``
     - ``--vllm.num_engines``
   * - ``--vllm_tensor_parallel_size``
     - ``--vllm.tensor_parallel_size``
   * - ``--vllm_sync_backend``
     - ``--vllm.sync_backend``
   * - ``--vllm_sync_with_ray``
     - ``--vllm.sync_with_ray``
   * - ``--vllm_gpu_memory_utilization``
     - ``--vllm.gpu_memory_utilization``
   * - ``--enforce_eager``
     - ``--vllm.enforce_eager``
   * - ``--enable_prefix_caching``
     - ``--vllm.enable_prefix_caching``
   * - ``--vllm_enable_sleep``
     - ``--vllm.enable_sleep``
   * - ``--deepspeed_enable_sleep``
     - ``--ds.enable_sleep``
   * - ``--zero_stage``
     - ``--ds.zero_stage``
   * - ``--param_dtype``
     - ``--ds.param_dtype``
   * - ``--adam_offload``
     - ``--ds.adam_offload``
   * - ``--zpg`` / ``--overlap_comm`` / ``--grad_accum_dtype`` / ``--deepcompile``
     - ``--ds.zpg`` / ``--ds.overlap_comm`` / ``--ds.grad_accum_dtype`` / ``--ds.deepcompile``
   * - ``--use_universal_ckpt``
     - ``--ds.use_universal_ckpt``
   * - ``--ds_tensor_parallel_size``
     - ``--ds.tensor_parallel_size``
   * - ``--ring_attn_size`` / ``--ring_head_stride``
     - ``--ds.ring_attn_size`` / ``--ds.ring_attn_head_stride``

Rollout / data / train / eval / ckpt / logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--temperature`` / ``--top_p`` / ``--max_new_tokens``
     - ``--rollout.temperature`` / ``--rollout.top_p`` / ``--rollout.max_new_tokens``
   * - ``--rollout_batch_size``
     - ``--rollout.batch_size``
   * - ``--micro_rollout_batch_size``
     - ``--rollout.micro_batch_size``
   * - ``--n_samples_per_prompt``
     - ``--rollout.n_samples_per_prompt``
   * - ``--vllm_generate_batch_size``
     - ``--rollout.vllm_generate_batch_size``
   * - ``--rollout_max_tokens_per_gpu``
     - ``--rollout.max_tokens_per_gpu``
   * - ``--prompt_data``
     - ``--data.prompt_dataset``  *(value is a dataset path, not a single prompt)*
   * - ``--prompt_data_probs`` / ``--prompt_split``
     - ``--data.prompt_probs`` / ``--data.prompt_split``
   * - ``--dataset`` / ``--dataset_probs`` / ``--dataset_split`` / ``--train_split``
     - ``--data.dataset`` / ``--data.dataset_probs`` / ``--data.dataset_split``
       *(``--train_split`` was a duplicate; removed.)*
   * - ``--input_key`` / ``--output_key`` / ``--label_key`` / ``--image_key``
     - ``--data.input_key`` / ``--data.output_key`` / ``--data.label_key`` / ``--data.image_key``
   * - ``--prompt_key`` / ``--chosen_key`` / ``--rejected_key``
     - ``--data.prompt_key`` / ``--data.chosen_key`` / ``--data.rejected_key``
   * - ``--input_template`` / ``--apply_chat_template`` / ``--tokenizer_chat_template``
     - ``--data.input_template`` / ``--data.apply_chat_template`` / ``--data.tokenizer_chat_template``
   * - ``--max_len`` / ``--max_samples`` / ``--multiturn``
     - ``--data.max_len`` / ``--data.max_samples`` / ``--data.multiturn``
   * - ``--packing_samples``
     - ``--ds.packing_samples`` *(moved from ``--data.*`` so engine-level loaders can read it)*
   * - ``--max_images_per_prompt`` / ``--dataloader_num_workers`` / ``--disable_fast_tokenizer``
     - ``--data.max_images_per_prompt`` / ``--data.dataloader_num_workers`` / ``--data.disable_fast_tokenizer``
   * - ``--train_batch_size`` / ``--micro_train_batch_size``
     - ``--train.batch_size`` / ``--train.micro_batch_size``
   * - ``--train_max_tokens_per_gpu``
     - ``--train.max_tokens_per_gpu``
   * - ``--max_epochs`` / ``--num_episodes`` / ``--seed``
     - ``--train.max_epochs`` / ``--train.num_episodes`` / ``--train.seed``
   * - ``--full_determinism``
     - ``--train.full_determinism_enable``
   * - ``--async_train`` / ``--async_queue_size`` / ``--partial_rollout``
     - ``--train.async_enable`` / ``--train.async_queue_size`` / ``--train.partial_rollout_enable``
   * - ``--use_dynamic_batch``
     - ``--train.dynamic_batch_enable``
   * - ``--enable_ema`` / ``--ema_beta``
     - ``--train.enable_ema`` / ``--train.ema_beta``
   * - ``--agent_func_path``
     - ``--train.agent_func_path``
   * - ``--eval_dataset`` / ``--eval_split`` / ``--eval_steps`` / ``--eval_temperature`` / ``--eval_n_samples_per_prompt``
     - ``--eval.dataset`` / ``--eval.split`` / ``--eval.steps`` / ``--eval.temperature`` / ``--eval.n_samples_per_prompt``
   * - ``--save_path`` / ``--ckpt_path`` / ``--save_steps``
     - ``--ckpt.output_dir`` / ``--ckpt.path`` / ``--ckpt.save_steps``
   * - ``--save_hf_ckpt`` / ``--disable_ds_ckpt`` / ``--max_ckpt_num`` / ``--max_ckpt_mem``
     - ``--ckpt.save_hf`` / ``--ckpt.disable_ds`` / ``--ckpt.max_num`` / ``--ckpt.max_mem``
   * - ``--load_checkpoint``
     - ``--ckpt.load_enable``
   * - ``--best_metric_key``
     - ``--ckpt.best_metric_key``
   * - ``--use_wandb``
     - ``--logger.wandb.key``
   * - ``--wandb_org`` / ``--wandb_group`` / ``--wandb_project`` / ``--wandb_run_name``
     - ``--logger.wandb.org`` / ``...group`` / ``...project`` / ``...run_name``
   * - ``--use_tensorboard`` / ``--logging_steps``
     - ``--logger.tensorboard_dir`` / ``--logger.logging_steps``

RL algorithm / loss
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--advantage_estimator``
     - ``--algo.advantage.estimator``
   * - ``--gamma`` / ``--lambd``
     - ``--algo.advantage.gamma`` / ``--algo.advantage.lambd``
   * - ``--no_advantage_std_norm``
     - ``--algo.advantage.no_std_norm``
   * - ``--enable_vllm_is_correction``
     - ``--algo.advantage.is_correction_enable``
   * - ``--vllm_is_correction_type``
     - ``--algo.advantage.is_correction_type``
   * - ``--vllm_is_truncated_threshold``
     - ``--algo.advantage.is_correction_threshold``
   * - ``--init_kl_coef`` / ``--kl_target`` / ``--kl_horizon`` / ``--kl_estimator`` / ``--use_kl_loss``
     - ``--algo.kl.init_coef`` / ``...target`` / ``...horizon`` / ``...estimator`` / ``...use_loss``
   * - ``--dynamic_filtering``
     - ``--algo.dynamic_filtering_enable``
   * - ``--dynamic_filtering_reward_range``
     - ``--algo.dynamic_filtering_range``
   * - ``--eps_clip`` / ``--eps_clip_low_high`` / ``--dual_clip``
     - ``--actor.eps_clip`` / ``--actor.eps_clip_low_high`` / ``--actor.dual_clip``
   * - ``--policy_loss_type``
     - ``--actor.policy_loss_type``
   * - ``--entropy_loss_coef``
     - ``--actor.entropy_coef``
   * - ``--aux_loss_coef``
     - ``--actor.aux_loss_coef`` *(PPO)* / ``--model.aux_loss_coef`` *(SFT/RM/DPO)*
   * - ``--freezing_actor_steps``
     - ``--critic.freezing_steps``  *(paired with critic warm-up)*
   * - ``--value_clip`` / ``--save_value_network``
     - ``--critic.value_clip`` / ``--critic.save_value_network``
   * - ``--ref_offload`` / ``--ref_reward_offload``
     - ``--ref.offload`` / ``--reward.offload``
   * - ``--normalize_reward``
     - ``--reward.normalize_enable``
   * - ``--reward_clip_range``
     - ``--reward.clip_range``
   * - ``--overlong_buffer_len`` / ``--overlong_penalty_factor``
     - ``--reward.overlong_buffer_len`` / ``--reward.overlong_penalty_factor``
   * - ``--stop_properly_penalty_coef``
     - ``--reward.stop_properly_penalty_coef``
   * - ``--value_head_prefix`` *(PPO reward / RM training / serve_rm)*
     - ``--ds.value_head_prefix``

Optimizer / scheduler
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--learning_rate``
     - ``--adam.lr``
   * - ``--actor_learning_rate`` / ``--critic_learning_rate``
     - ``--actor.adam.lr`` / ``--critic.adam.lr``
   * - ``--adam_betas`` / ``--adam_eps``
     - ``--adam.betas`` / ``--adam.eps`` (per-entity in PPO)
   * - ``--l2``
     - ``--adam.weight_decay`` (per-entity in PPO; under Muon, DS stamps the same value
       into both Muon and aux-Adam groups — there is no separate ``--muon.weight_decay``)
   * - ``--muon_lr`` / ``--muon_momentum``
     - ``--muon.lr`` / ``--muon.momentum``
   * - ``--muon_ns_steps`` / ``--muon_nesterov`` / ``--no_muon_nesterov``
     - ``--muon.ns_steps`` / ``--muon.nesterov`` / ``--muon.no_nesterov``
   * - ``--muon_adam_lr``
     - *removed* — aux-Adam subgroup reuses ``--adam.lr``
   * - ``--lr_scheduler`` / ``--lr_warmup_ratio`` / ``--min_lr_ratio`` / ``--max_norm``
     - flat in SFT / RM / DPO; ``--actor.*`` / ``--critic.*`` in PPO

SFT / RM / DPO model-level loss knobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Old
     - New
   * - ``--beta`` *(DPO)*
     - ``--model.beta``
   * - ``--ipo``
     - ``--model.ipo_enable``
   * - ``--label_smoothing`` / ``--nll_loss_coef``
     - ``--model.label_smoothing`` / ``--model.nll_loss_coef``
   * - ``--compute_fp32_loss`` / ``--margin_loss``
     - ``--model.compute_fp32_loss_enable`` / ``--model.margin_loss_enable``
   * - ``--pretrain_mode`` *(SFT)*
     - ``--model.pretrain_mode_enable``
   * - ``--loss`` *(RM)*
     - ``--model.loss_type``

Removed flags
~~~~~~~~~~~~~

These were dead code or duplicates in earlier releases and have been deleted:

- ``--actor_init_on_gpu`` (legacy init-on-GPU path; no longer needed).
- ``--ptx_coef`` / ``--actor.ptx_coef`` (PPO-PTX was never wired up in the trainer).
- ``--train.perf_tracking_enable`` (no corresponding implementation).
- ``--train_split`` (duplicate of ``--data.dataset_split``).

Common CLI Options
==================

This page collects CLI options shared across OpenRLHF trainers (SFT / RM / DPO / RL). For trainer-specific knobs see :doc:`non_rl` or :doc:`agent_training`. End-to-end launch examples live under `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

.. contents::
   :local:
   :depth: 2

Training
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--zero_stage``
     - DeepSpeed ZeRO stage (0 / 1 / 2 / 3). Use 3 for large models.
   * - ``--ds_tensor_parallel_size``
     - DeepSpeed tensor parallelism (AutoTP) size — only for ``--zero_stage 0/1/2``.
   * - ``--adam_offload``
     - Offload the Adam optimizer state to CPU; saves GPU memory at the cost of step time.
   * - ``--adam_betas``
     - Adam β coefficients (default ``(0.9, 0.95)``).
   * - ``--overlap_comm``
     - Overlap backward and gradient comm in DeepSpeed (uses 4.5× the default bucket sizes — needs more memory).
   * - ``--param_dtype``
     - Parameter dtype: ``bf16`` (default) or ``fp16``.
   * - ``--attn_implementation``
     - Attention backend: ``eager`` / ``flash_attention_2`` / ``flash_attention_3`` / ``kernels-community/vllm-flash-attn3``.
   * - ``--gradient_checkpointing``
     - Trade compute for memory (recompute activations).
   * - ``--save_path``
     - Final HuggingFace-format model save path.
   * - ``--learning_rate`` / ``--actor_learning_rate`` / ``--critic_learning_rate``
     - Learning rates (the latter two are PPO-specific).
   * - ``--lr_scheduler``
     - LR scheduler (default ``cosine_with_min_lr``).
   * - ``--lr_warmup_ratio``
     - Warmup ratio of total steps.
   * - ``--l2``
     - Weight decay coefficient.
   * - ``--max_norm``
     - Gradient clipping (default ``1.0``).
   * - ``--micro_train_batch_size`` / ``--train_batch_size``
     - Per-GPU and global training batch sizes.
   * - ``--max_epochs``
     - Number of training epochs.
   * - ``--aux_loss_coef``
     - MoE balancing loss coefficient (set ``> 0`` for MoE models).
   * - ``--use_liger_kernel``
     - Use `Liger Kernel <https://github.com/linkedin/Liger-Kernel>`_ for fused ops.
   * - ``--deepcompile``
     - Enable `DeepCompile <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md>`_ graph compilation (PyTorch 2.0+).
   * - ``--seed``
     - Global random seed (default ``42``).
   * - ``--full_determinism``
     - Bit-reproducible behavior across runs (slower; uses vLLM v1 + fixed seed paths).

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--dataset`` / ``--prompt_data``
     - Dataset name(s) or path(s) for training (RL trainers use ``--prompt_data``).
   * - ``--dataset_probs`` / ``--prompt_data_probs``
     - Mixing probabilities for multiple datasets (e.g., ``0.1,0.4,0.5``).
   * - ``--eval_dataset``
     - Dataset name(s) or path(s) for evaluation.
   * - ``--input_key`` / ``--output_key`` / ``--label_key`` / ``--chosen_key`` / ``--rejected_key`` / ``--image_key``
     - JSON keys for input / output / label / preference pairs / image paths. Vary by trainer.
   * - ``--apply_chat_template``
     - Use HuggingFace ``tokenizer.apply_chat_template``.
   * - ``--tokenizer_chat_template``
     - Override the tokenizer's default chat template.
   * - ``--input_template``
     - Custom Python format string when not using a chat template (e.g., ``$'User: {}\\nAssistant: '``).
   * - ``--max_len``
     - Max total sequence length (prompt + response).
   * - ``--max_new_tokens`` *(RL only)*
     - Max generation tokens. If unset, dynamically computed as ``max_len - prompt_len`` per sample.
   * - ``--max_samples``
     - Cap on training samples.
   * - ``--packing_samples``
     - Pack multiple samples per sequence (Flash-Attention path) — large speedup, removes padding waste.
   * - ``--dataloader_num_workers``
     - Number of DataLoader worker processes (default ``0``).

LoRA / QLoRA
------------

LoRA / QLoRA is supported by SFT / RM / DPO. **Not supported** by Ray + vLLM PPO.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--lora_rank``
     - Set to an integer ``> 0`` to enable LoRA (rank of the low-rank update).
   * - ``--lora_alpha``
     - LoRA alpha scaling.
   * - ``--lora_dropout``
     - LoRA dropout for HuggingFace PEFT.
   * - ``--target_modules``
     - PEFT target modules (e.g., ``q_proj k_proj v_proj o_proj``).
   * - ``--load_in_4bit``
     - Load the base model in 4-bit (QLoRA).

When LoRA is enabled, only the adapter is saved. To deploy or continue training, merge the adapter back with the base model:

.. code-block:: bash

   python -m openrlhf.cli.lora_combiner \
      --model_path meta-llama/Meta-Llama-3-8B \
      --lora_path ./checkpoint/llama3-8b-rm \
      --output_path ./checkpoint/llama-3-8b-rm-combined \
      --is_rm \
      --param_dtype bf16

Use ``--is_rm`` when merging a reward model (preserves the score head).

Logging & monitoring
--------------------

- ``--use_wandb {token-or-True}``: Wandb logging (or ``True`` if you've already run ``wandb login``).
- ``--wandb_project`` / ``--wandb_group`` / ``--wandb_run_name``: Wandb metadata.
- ``--use_tensorboard {logdir}``: TensorBoard logging.
- ``--logging_steps``: log every *N* training steps.
- ``--eval_steps``: evaluate every *N* steps (``-1`` to disable).

Long context & checkpointing
----------------------------

These have their own dedicated pages:

- **RingAttention** (``--ring_attn_size`` / ``--ring_head_stride``) — see :doc:`sequence_parallelism`.
- **Checkpointing** (``--save_steps`` / ``--ckpt_path`` / ``--load_checkpoint`` /
  ``--save_hf_ckpt`` / ``--best_metric_key`` / ``--enable_ema`` …) — see :doc:`checkpoint`.

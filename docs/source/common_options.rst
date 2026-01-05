Common Options (Shared Across Trainers)
==================================================

This chapter collects CLI options that are shared across OpenRLHF trainers (SFT/RM/DPO/RL, etc.).

For end-to-end launch examples, also see `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

Training
--------

- ``--zero_stage``: DeepSpeed ZeRO Stage
- ``--adam_offload``: Offload the Adam Optimizer to CPU
- ``--adam_betas``: Adam betas, default value is ``(0.9, 0.95)``
- ``--overlap_comm``: Enable backward & gradient overlap_comm for Deepspeed (overlap_comm uses 4.5x the allgather_bucket_size and reduce_bucket_size values.)
- ``--bf16``: Enable bfloat16
- ``--attn_implementation``: Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)
- ``--gradient_checkpointing``: Enable Gradient Checkpointing
- ``--save_path``: Final HuggingFace model save path
- ``--use_wandb``: Set to ``{wandb_token}`` or ``True`` with shell command ``wandb login``
- ``--use_tensorboard``: Set to ``{tensorboard logs path}``
- ``--learning_rate``: Learning Rate
- ``--l2``: Weight Decay
- ``--lr_scheduler``: Learning Rate Scheduler
- ``--max_norm``: Gradient clipping
- ``--micro_train_batch_size``: Batch size per GPU for training
- ``--train_batch_size``: Global training batch size
- ``--aux_loss_coef``: Balancing loss coefficient for MoE
- ``--max_epoch``: Training epochs
- ``--lr_warmup_ratio``: Warmup ratio of the learning rate
- ``--use_liger_kernel``: Use Liger Kernel
- ``--ds_tensor_parallel_size``: DeepSpeed Tensor Parallel Size (AutoTP), only used when ``--zero_stage 0 / 1 / 2``

Datasets
--------

- ``--dataset``: Dataset names or paths for training
- ``--dataset_probs``: Dataset mixing probabilities
- ``--eval_dataset``: Dataset names or paths for evaluation
- ``--input_key``: Input JSON key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--input_template``: Custom ``input_template`` (when not using ``tokenizer.apply_chat_template``), set to ``None`` to disable it. Such as ``$'User: {}\\nAssistant: '``.
- ``--max_len``: Max length for the samples
- ``--max_samples``: Max training samples
- ``--packing_samples``: Packing samples using Flash Attention 2

LoRA
----

- ``--load_in_4bit``: Use QLoRA
- ``--lora_rank``: Set to ``integer > 0`` to enable LoRA
- ``--lora_dropout``: LoRA dropout for HuggingFace PEFT (LoRA)
- ``--target_modules``: Target modules for HuggingFace PEFT (LoRA)

If you use ``LoRA (Low-Rank Adaptation)``, OpenRLHF will not save the full weights by default instead of ``LoRA Adapter``. To continue in your task normally, you should combine the ``Adapter`` with weights of your base model:

.. code-block:: bash

   python -m openrlhf.cli.lora_combiner \
      --model_path meta-llama/Meta-Llama-3-8B \
      --lora_path ./checkpoint/llama3-8b-rm \
      --output_path ./checkpoint/llama-3-8b-rm-combined \
      --is_rm \
      --bf16



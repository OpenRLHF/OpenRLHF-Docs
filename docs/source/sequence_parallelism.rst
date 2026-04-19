Sequence Parallelism (RingAttention)
====================================

OpenRLHF supports long-context training through **RingAttention** — a sequence-parallel attention
algorithm that distributes a single long sequence across multiple GPUs while overlapping KV-block
communication with blockwise attention compute. References: `RingAttention paper
<https://arxiv.org/abs/2310.01889>`_ and `ring-flash-attention
<https://github.com/zhuzilin/ring-flash-attention>`_.

When to use
-----------

Enable RingAttention when:

- Your sequence length exceeds what a single GPU can hold (typically ``> 8K`` tokens with bf16 +
  flash-attn).
- You want to train with longer max contexts without dropping batch size.

For shorter sequences, RingAttention adds communication overhead — leave it off
(``--ds.ring_attn_size 1``).

Installation
------------

.. code-block:: bash

   pip install ring_flash_attn
   # or install from source:
   pip install git+https://github.com/zhuzilin/ring-flash-attention

Or install OpenRLHF with the ``ring`` extra: ``pip install openrlhf[vllm,ring,liger]``.

Flags
-----

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--ds.ring_attn_size``
     - Sequence-parallel group size. The sequence is split into this many chunks, one per GPU in
       the group.
   * - ``--ds.ring_attn_head_stride``
     - Number of attention heads processed per RingAttention round (must divide
       ``num_attention_heads``). Larger stride → faster (fewer rounds) but more memory; smaller
       stride → slower but lower peak memory.

A common starting point: ``--ds.ring_attn_size 2 --ds.ring_attn_head_stride 2``. Increase
``ring_attn_size`` (e.g., 4 or 8) for very long contexts; tune ``ring_attn_head_stride`` based on
memory headroom.

.. note::
   ``--ds.ring_attn_size > 1`` requires ``--ds.packing_samples`` (the trainer auto-enables it
   with a warning if missing).

Examples
--------

SFT
~~~

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
      --ds.ring_attn_size 2 \
      --ds.ring_attn_head_stride 2 \
      --ckpt.output_dir ./checkpoint/llama3-8b-sft \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1

DPO
~~~

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
      --ds.ring_attn_size 2 \
      --ds.ring_attn_head_stride 2 \
      --ckpt.output_dir ./checkpoint/llama3-8b-ring-dpo \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --ckpt.load_enable

Reward Model
~~~~~~~~~~~~

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
      --train.micro_batch_size 1 \
      --train.max_epochs 1 \
      --adam.lr 9e-6 \
      --ds.zero_stage 3 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --model.gradient_checkpointing_enable \
      --ds.ring_attn_size 2 \
      --ds.ring_attn_head_stride 2 \
      --ckpt.output_dir ./checkpoint/llama3-8b-rm \
      --ckpt.save_steps -1 \
      --logger.logging_steps 1 \
      --eval.steps -1 \
      --ckpt.load_enable

PPO / GRPO / REINFORCE++
~~~~~~~~~~~~~~~~~~~~~~~~

The Ray + vLLM RL trainers support the same flags. See `train_ppo_ray_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_ray_hybrid_engine.sh>`_
— uncomment the ``--ds.ring_attn_size`` and ``--ds.ring_attn_head_stride`` lines at the bottom.

.. tip::
   When pairing RingAttention with ``--ds.packing_samples``, packing happens **before** the
   sequence is sharded — your ``--train.micro_batch_size`` controls how many *packed* sequences
   each ring sees, not how many original samples.

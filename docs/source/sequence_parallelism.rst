Sequence Parallelism (RingAttention)
====================================

OpenRLHF supports long-context training through **RingAttention** — a sequence-parallel attention algorithm that distributes a single long sequence across multiple GPUs while overlapping KV-block communication with blockwise attention compute. References: `RingAttention paper <https://arxiv.org/abs/2310.01889>`_ and `ring-flash-attention <https://github.com/zhuzilin/ring-flash-attention>`_.

When to use
-----------

Enable RingAttention when:

- Your sequence length exceeds what a single GPU can hold (typically ``> 8K`` tokens with bf16 + flash-attn).
- You want to train with longer max contexts without dropping batch size.

For shorter sequences, RingAttention adds communication overhead — leave it off (``--ring_attn_size 1``).

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
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--ring_attn_size``
     - Sequence-parallel group size. The sequence is split into this many chunks, one per GPU in the group.
   * - ``--ring_head_stride``
     - Number of attention heads processed per RingAttention round (must divide ``num_attention_heads``). Larger stride → faster (fewer rounds) but more memory; smaller stride → slower but lower peak memory.

A common starting point: ``--ring_attn_size 2 --ring_head_stride 2``. Increase ``ring_attn_size`` (e.g., 4 or 8) for very long contexts; tune ``ring_head_stride`` based on memory headroom.

Examples
--------

SFT
~~~

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question --output_key response \
      --input_template $'User: {}\nAssistant: ' \
      --train_batch_size 256 --micro_train_batch_size 8 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --zero_stage 2 --max_epochs 1 --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --packing_samples \
      --ring_attn_size 2 --ring_head_stride 2 \
      --learning_rate 5e-6 --gradient_checkpointing

DPO
~~~

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_dpo \
      --save_path ./checkpoint/llama3-8b-ring-dpo \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --train_batch_size 256 --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 --max_epochs 1 --max_len 8192 \
      --zero_stage 3 --learning_rate 5e-7 --beta 0.1 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template --chosen_key chosen --rejected_key rejected \
      --ring_attn_size 2 --ring_head_stride 2 \
      --packing_samples --attn_implementation flash_attention_2 \
      --load_checkpoint --gradient_checkpointing

Reward Model
~~~~~~~~~~~~

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --train_batch_size 256 --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 --max_epochs 1 --max_len 8192 \
      --zero_stage 3 --learning_rate 9e-6 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template --chosen_key chosen --rejected_key rejected \
      --ring_attn_size 2 --ring_head_stride 2 \
      --packing_samples --attn_implementation flash_attention_2 \
      --load_checkpoint --gradient_checkpointing

PPO / GRPO / REINFORCE++
~~~~~~~~~~~~~~~~~~~~~~~~

The Ray + vLLM RL trainers support the same flags. See `train_ppo_ray_hybrid_engine.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_ray_hybrid_engine.sh>`_ — uncomment the ``--ring_attn_size`` and ``--ring_head_stride`` lines at the bottom.

.. tip::
   When pairing RingAttention with ``--packing_samples``, packing happens **before** the sequence is sharded — your ``micro_train_batch_size`` controls how many *packed* sequences each ring sees, not how many original samples.

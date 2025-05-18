Sequence Parallelism
=====

Ring Attention
------------

OpenRLHF supports long-text model training based on RingAttention.
Ring Attention with Blockwise Transformers (Ring Attention) leverages blockwise computation of self-attention and feedforward to distribute long sequences across multiple devices while fully overlapping the communication of key-value blocks with the computation of blockwise attention. 
More details are in `<https://arxiv.org/abs/2310.01889>`_ and `<https://github.com/zhuzilin/ring-flash-attention>`_. 


Examples
------------

First, pip install ``ring_flash_attn``.

.. code-block:: bash
   
   pip install ring_flash_attn
   # or install from source
   pip install git+https://github.com/zhuzilin/ring-flash-attention

Then run the training scripts

Related options:

- ``--ring_attn_size``: Ring attention group size
- ``--ring_head_stride``: the number of heads to do ring attention each time. It should be a divisor of the number of heads. A larger value may results in faster training but will consume more memory.

SFT


.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\nAssistant: ' \
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
      --bf16 \
      --flash_attn \
      --packing_samples \
      --ring_attn_size 2 \
      --ring_head_stride 2 \
      --learning_rate 5e-6 \
      --gradient_checkpointing \

DPO

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_dpo \
      --save_path ./checkpoint/llama3-8b-ring-dpo \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 5e-7 \
      --beta 0.1 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template \
      --chosen_key chosen \
      --rejected_key rejected \
      --ring_attn_size 2 \
      --ring_head_stride 2 \
      --packing_samples \
      --flash_attn \
      --load_checkpoint \
      --gradient_checkpointing

RM Training

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 9e-6 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template \
      --chosen_key chosen \
      --rejected_key rejected \
      --ring_attn_size 2 \
      --ring_head_stride 2 \
      --packing_samples \
      --flash_attn \
      --load_checkpoint \
      --gradient_checkpointing


The PPO/GRPO/REINFORCE++ also support sequence parallelism using the same options.
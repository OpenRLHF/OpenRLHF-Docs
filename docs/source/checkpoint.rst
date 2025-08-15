Checkpoint
=====

How to save training checkpoints?
------------

Since training large models is time-consuming and expensive, reloading checkpoints (including model/optimizer/scheduler states and training dataset progress) becomes crucial when the training crashes. 
OpenRLHF has re-implemented the `DistributedSampler <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_sampler.py>`_ with resumable training dataset progress, and the checkpoint mechanism based on DeepSpeed's checkpoint API. 

The examples are as follows,

Related options:

- ``--save_steps``: Number of ``global training steps`` between saving checkpoints. For PPO, it refers to the number of model updates (excluding mini-batches).
- ``--ckpt_path``: Directory path where checkpoints will be saved.
- ``--load_checkpoint``: Load checkpoint for resuming training (Skip if the checkpoint does not exist).
- ``--save_hf_ckpt``: Save huggingfae models for each checkpoint.
- ``--disable_ds_ckpt``: Do not save DeepSpeed checkpoints to save disk space, but this will prevent the training progress from being recoverable.
- ``--max_ckpt_num``: Maximum number of latest checkpoints to keep.
- ``--max_ckpt_mem``: Maximum memory size (GB) allocated for storing checkpoints.
- ``--use_ds_universal_ckpt``: Use deepspeed universal checkpoint.

SFT

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\nAssistant: ' \
      --train_batch_size 256 \
      --micro_train_batch_size 2 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --zero_stage 2 \
      --max_epochs 1 \
      --bf16 \
      --attn_implementation flash_attention_2 \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --save_steps 200 \
      --ckpt_path ./ckpt \
      --save_hf_ckpt \
      --load_checkpoint \
      --use_wandb {wandb_token}
      

Ray PPO with vLLM

.. code-block:: bash
   
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 2 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 2 \
      --critic_num_nodes 1 \
      --critic_num_gpus_per_node 2 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 2 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 2 \
      --colocate_critic_reward \
      --colocate_actor_ref \
      --ref_reward_offload \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
      --micro_train_batch_size 8 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 16 \
      --rollout_batch_size 1024 \
      --max_samples 100000 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --generate_max_len 1024 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --normalize_reward \
      --adam_offload \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --save_steps 20 \
      --ckpt_path /openrlhf/examples/checkpoint/ckpt/ \
      --save_hf_ckpt \
      --load_checkpoint \
      --use_wandb {wandb_token}

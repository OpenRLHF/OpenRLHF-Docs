Usage
=====

Common Options
---------------

We provide launch scripts for supported algorithms in the `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_ directory.
Here are some common options for the supported algorithms.

Training

- ``--zero_stage``: DeepSpeed ZeRO Stage
- ``--adam_offload``: Offload the Adam Optimizer to GPU
- ``--adam_betas``: Adam betas, default value is (0.9, 0.95)
- ``--bf16``: Enable bfloat16
- ``--flash_attn``: Enable Flash Attention 2
- ``--gradient_checkpointing``: Enable Gradient Checkpointing
- ``--save_path``: Final HuggingFace model save path
- ``--use_wandb``: Set to ``{wandb_token}`` or ``True`` with shell command ``wandb login``
- ``--learning_rate``: Learning Rate
- ``--l2``: Weight Decay
- ``--lr_scheduler``: Learning Rate Scheduler 
- ``--max_norm``: Gradient clipping
- ``--micro_train_batch_size``: Batch size per GPU for training
- ``--train_batch_size``: Global training batch size
- ``--aux_loss_coef``: Balancing loss coefficient for MoE
- ``--max_epoch``: Training epochs

Datasets

- ``--dataset``: Dataset names or paths
- ``--dataset_probs``: Dataset mixing probabilities
- ``--input_key``: Input JSON key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--input_template``: Custom ``input_template`` (when not using ``tokenizer.apply_chat_template``), set to ``None`` to disable it. Such as ``'User: {}\nAssistant: '``.
- ``--max_len``: Max length for the samples
- ``--max_samples``: Max training samples
- ``--train_split``: HF datasets split for training, default value is ``train``
- ``--eval_split``: HF datasets split for evaluation, default value is ``test``
- ``--packing_samples``: Packing samples using Flash Attention 2

LoRA

- ``--load_in_4bit``: Use QLoRA
- ``--lora_rank``: Set to ``integer > 0`` to enable LoRA
- ``--lora_dropout``: LoRA dropout for HuggingFace PEFT (LoRA)
- ``--target_modules``: Target modules for HuggingFace PEFT (LoRA)


Supervised Fine-tuning
----------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template 'User: {}\nAssistant: ' \
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
      --flash_attn \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: JSON dataset key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--pretrain_mode``: Continue pretrain mode
- ``--packing_samples``: Packing SFT samples

.. note:: OpenRLHF SFT/DPO/RM trainers supports ``--packing_samples`` `using --flash_attn <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_



Reward Model Training
---------------------

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
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--chosen_key`` JSON dataset key for chosen conversions
- ``--rejected_key`` JSON dataset key for rejected conversions
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--value_head_prefix``: custom ``value_head`` (score head) prefix
- ``--packing_samples``: Packing RM samples


.. _rayppo:

PPO with Ray (vLLM)
------------

To improve RLHF training speed or support 70B models, we can use the ``PPO with Ray and vLLM acceleration``

.. code-block:: bash
   
   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # if you want to launch ray on more nodes, use
   ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

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
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}


.. note:: Do not set `--vllm_num_engines` means not using the vLLM engine. Ray + vLLM does not supports LoRA currently.
You can also use ``setup_commands`` to let Ray automatically deploy the environment, such as ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'``

Options

Ray and vLLM

- ``--ref_num_nodes``: Number of nodes for the Reference Model
- ``--ref_num_gpus_per_node``: Number of GPUs per node for the Reference Model
- ``--reward_num_nodes``: Number of nodes for the Reward Model
- ``--reward_num_gpus_per_node``: Number of GPUs per node for the Reward Model
- ``--critic_num_nodes``: Number of nodes for the Critic Model
- ``--critic_num_gpus_per_node``: Number of GPUs per node for the Critic Model
- ``--actor_num_nodes``: Number of nodes for the Actor Model
- ``--actor_num_gpus_per_node``: Number of GPUs per node for the Actor Model
- ``--vllm_num_engines``: Number of vLLM engines, set to 0 to disable vLLM
- ``--vllm_tensor_parallel_size``: Tensor Parallel Size for vLLM engines
- ``--colocate_critic_reward``: Colocate Critic and Reward nodes. Ensure that the GPU configurations for Critic and Reward are identical
- ``--colocate_actor_ref``: Colocate Actor and Reference Model nodes. Ensure that the GPU configurations for Actor and Ref are identical
- ``--ref_reward_offload``: Offload Reward and Reference models to GPU
- ``--vllm_sync_backend``: Set to ``nccl`` or ``gloo`` for vLLM weights sync
- ``--enable_prefix_caching``: Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation

PPO

- ``--save_value_network``: Save the Value Network after training is complete
- ``--normalize_reward``: Enable Reward Normalization
- ``--value_head_prefix``: custom ``value_head``  (score head) prefix for the reward model
- ``--init_kl_coef``: KL penalty coeff
- ``--max_epochs``: Number of PPO training epochs
- ``--micro_train_batch_size``: Batch size per GPU for training
- ``--train_batch_size``: PPO mini-batch size
- ``--micro_rollout_batch_size``: Batch size per GPU for generation
- ``--rollout_batch_size``: Replay Buffer Size
- ``--prompt_max_len``: Max length for the prompts
- ``--generate_max_len``: Max length for the responses
- ``--n_samples_per_prompt``: Generate n samples for each promot
- ``--freezing_actor_steps``: Freezing the actor parameters to init critic in the first n steps
- ``--reward_pretrain``: Can be set to multiple reward models, such as ``RewardMode1,RewardModel2,RewardModel3``
- ``--actor_learning_rate``: Actor model learning rate
- ``--critic_learning_rate``: Critic model learning rate

Datasets

- ``--prompt_data``: Dataset names or paths (Prompts)
- ``--prompt_data_probs``: Dataset mixing probabilities
- ``--pretrain_data``: Dataset names or paths (Pretrain)
- ``--pretrain_data_probs``: Dataset mixing probabilities
- ``--prompt_split``: HF datasets split for training (Prompts), default value is ``train``
- ``--pretrain_split``: HF datasets split for training (Pretrain), default value is ``train`` 


Direct Preference Optimization (DPO)
-----------------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_dpo \
      --save_path ./checkpoint/llama3-8b-dpo \
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
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}


Options

- ``--chosen_key`` JSON dataset key for chosen conversions
- ``--rejected_key`` JSON dataset key for rejected conversions
- ``--ref_offload`` Offload Reference Model to CPU
- ``--beta`` The beta factor in DPO loss. Higher beta means less divergence from the initial policy. 
- ``--ipo`` for IPO loss. 
- ``--label_smoothing`` for cDPO loss. 
- ``--packing_samples``: Packing DPO samples


Kahneman-Tversky Optimization (KTO)
------------------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_kto \
      --save_path ./checkpoint/llama3-8b-kto \
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
      --dataset Dylan2048/ultrafeedback-unpaired-preferences \
      --input_key instruction \
      --output_key response \
      --label_key score \
      --input_template 'User: {}\nAssistant: ' \
      --flash_attn \
      --beta 0.1 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: JSON dataset key for the instructions
- ``--output_key``: JSON dataset key for the responses
- ``--label_key``: JSON dataset key for the label
- ``--ref_offload``: Offload Reference Model to CPU
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template`` (Use ``--input_key`` to specify ``conversions``)


Rejection Sampling & RAFT
-------------------------

.. code-block:: bash

   checkSuccess() {
      if [[ $? != 0 ]]; then
         echo "FAILED $1"
         exit 1
      fi
   }

   mkdir -p ./checkpoint/llama-3-8b-rejection
   GENERATE_OUTPUT=./checkpoint/llama-3-8b-rejection/generate.jsonl
   RM_OUTPUT=./checkpoint/llama-3-8b-rejection/rm.jsonl
   ITER_LOG_PATH=./checkpoint/llama-3-8b-rejection/iter.log
   MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-rejection

   TRAINING_ITERS=10
   ROLLOUT_BATCH_SIZE=10240

   POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture

   iter=0
   if [ -f $ITER_LOG_PATH ]; then
      iter=$(cat $ITER_LOG_PATH)
   fi

   while (($iter < $TRAINING_ITERS)); do
      echo "Iter: $iter"
      # Use latest model if past first iteration
      if ((iter > 0)); then
         POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
      fi

      read -r -d '' generate_commands <<EOF
   openrlhf.cli.batch_inference \
      --eval_task generate_vllm \
      --pretrain $POLICY_MODEL_PATH \
      --bf16 \
      --max_new_tokens 2048 \
      --prompt_max_len 2048 \
      --dataset OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --temperature 0.9
      --best_of_n 4 \
      --enable_prefix_caching \
      --tp_size 4 \
      --micro_batch_size 64 \
      --iter $iter \
      --rollout_batch_size $ROLLOUT_BATCH_SIZE \
      --output_path $GENERATE_OUTPUT
   EOF
      echo $generate_commands
      python -m $generate_commands
      checkSuccess "GENERATE"

      read -r -d '' get_rewards_commands <<EOF
   openrlhf.cli.batch_inference \
      --eval_task rm \
      --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --bf16 \
      --max_len 2048 \
      --dataset $GENERATE_OUTPUT  \
      --dataset_probs 1.0 \
      --zero_stage 0 \
      --post_processor rs \
      --micro_batch_size 4 \
      --output_path $RM_OUTPUT
   EOF
      echo $get_rewards_commands
      deepspeed --module $get_rewards_commands
      checkSuccess "RM"

      read -r -d '' sft_commands <<EOF
   openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset $RM_OUTPUT \
      --dataset_probs 1.0 \
      --train_batch_size 128 \
      --micro_train_batch_size 2 \
      --pretrain $POLICY_MODEL_PATH \
      --save_path ./checkpoint/llama-3-8b-rejection \
      --input_template "" \
      --input_key input \
      --output_key output \
      --zero_stage 2 \
      --max_epochs 1 \
      --bf16 \
      --learning_rate 2e-6 \
      --gradient_checkpointing
   EOF
      echo $sft_commands
      deepspeed --module $sft_commands
      checkSuccess "SFT"

      iter=$((iter + 1))
      if [[ "$ITER_LOG_PATH" != "null" ]]; then
         echo $iter >$ITER_LOG_PATH
      fi
   done

.. _batch_inference:

Options for ``openrlhf.cli.batch_inference``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--eval_task``: set to ``generate_vllm``, ``generate`` (HF generate) or ``rm``
- ``--iter``: used to slice the datasets in range ``iter * rollout_batch_size: (iter + 1) * rollout_batch_size``
- ``--rollout_batch_size``: number of samples to generate
- ``--best_of_n``: number of responses to generate per prompt
- ``--input_key``: JSON dataset key
- ``--tp_size``: TP Size for vLLM
- ``--enable_prefix_caching``: Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation
- ``--max_new_tokens``: Max new tokens in generation
- ``--prompt_max_len``: Max tokens for prompt
- ``--greedy_sampling``: Use Greedy sampling
- ``--top_p``: ``top_p`` for Sampling
- ``--temperature``:  ``temperature`` for Sampling
- ``--repetition_penalty``: ``repetition_penalty`` for Sampling
- ``--value_head_prefix``: ``value_head`` prefix for Reward Model
- ``--post_processor``: set to ``rs`` (Rejection Sampling), ``csft`` (Conditional SFT), ``iter_dpo`` (Iterative DPO) or ``None``


Iterative DPO
------------

.. code-block:: bash

   checkSuccess() {
      if [[ $? != 0 ]]; then
         echo "FAILED $1"
         exit 1
      fi
   }

   mkdir -p ./checkpoint/llama-3-8b-iter-dpo
   GENERATE_OUTPUT=./checkpoint/llama-3-8b-iter-dpo/generate.jsonl
   RM_OUTPUT=./checkpoint/llama-3-8b-iter-dpo/rm.jsonl
   MODEL_OUTPUT_PATH=./checkpoint/llama-3-8b-iter-dpo/checkpoint
   ITER_LOG_PATH=null

   TRAINING_ITERS=5
   ROLLOUT_BATCH_SIZE=10240

   POLICY_MODEL_PATH=OpenRLHF/Llama-3-8b-sft-mixture
   REF_MODEL_PATH=$POLICY_MODEL_PATH

   iter=0
   if [ -f $ITER_LOG_PATH ]; then
      iter=$(cat $ITER_LOG_PATH)
   fi

   while (($iter < $TRAINING_ITERS)); do
      echo "Iter: $iter"
      # Use latest model if past first iteration
      if ((iter > 0)); then
         POLICY_MODEL_PATH=$MODEL_OUTPUT_PATH
      fi

      read -r -d '' generate_commands <<EOF
   openrlhf.cli.batch_inference \
      --eval_task generate_vllm \
      --pretrain $POLICY_MODEL_PATH \
      --max_new_tokens 2048 \
      --prompt_max_len 2048 \
      --dataset OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --temperature 1.0 \
      --tp_size 4 \
      --best_of_n 16 \
      --enable_prefix_caching \
      --max_num_seqs 64 \
      --iter $iter \
      --rollout_batch_size $ROLLOUT_BATCH_SIZE \
      --output_path $GENERATE_OUTPUT
   EOF
      echo $generate_commands
      python -m $generate_commands
      checkSuccess "GENERATE"

      read -r -d '' get_rewards_commands <<EOF
   openrlhf.cli.batch_inference \
      --eval_task rm \
      --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --bf16 \
      --max_len 4096 \
      --dataset $GENERATE_OUTPUT  \
      --dataset_probs 1.0 \
      --zero_stage 0 \
      --post_processor iter_dpo \
      --micro_batch_size 4 \
      --output_path $RM_OUTPUT
   EOF
      echo $get_rewards_commands
      deepspeed --module $get_rewards_commands
      checkSuccess "RM"

      read -r -d '' dpo_commands <<EOF
   openrlhf.cli.train_dpo \
      --max_len 4096 \
      --dataset $RM_OUTPUT \
      --dataset_probs 1.0 \
      --train_batch_size 128 \
      --micro_train_batch_size 2 \
      --pretrain $POLICY_MODEL_PATH \
      --ref_pretrain $REF_MODEL_PATH \
      --save_path $MODEL_OUTPUT_PATH \
      --zero_stage 3 \
      --max_epochs 1 \
      --bf16 \
      --learning_rate 5e-7 \
      --gradient_checkpointing
   EOF
      echo $dpo_commands
      deepspeed --module $dpo_commands
      checkSuccess "DPO"

      iter=$((iter + 1))
      if [[ "$ITER_LOG_PATH" != "null" ]]; then
         echo $iter >$ITER_LOG_PATH
      fi
   done

Options for ``batch_inference``, refer to :ref:`batch_inference`.


Conditional SFT
------------

.. code-block:: bash

   checkSuccess() {
      if [[ $? != 0 ]]; then
         echo "FAILED $1"
         exit 1
      fi
   }

   RM_OUTPUT=./checkpoint/llama-2-8b-csft/rm.jsonl

   read -r -d '' get_rewards_commands <<EOF
   openrlhf.cli.batch_inference \
      --eval_task rm \
      --pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --bf16 \
      --max_len 4096 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --input_key chosen \
      --apply_chat_template \
      --max_samples 128000 \
      --zero_stage 0 \
      --post_processor csft \
      --normalize_reward
      --micro_batch_size 4 \
      --output_path $RM_OUTPUT
   EOF

   read -r -d '' sft_commands <<EOF
   openrlhf.cli.train_sft \
      --max_len 4096 \
      --dataset $RM_OUTPUT \
      --dataset_probs 1.0 \
      --train_batch_size 128 \
      --micro_train_batch_size 2 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --save_path ./checkpoint/llama-3-8b-csft \
      --zero_stage 2 \
      --max_epochs 1 \
      --bf16 \
      --learning_rate 5e-6 \
      --gradient_checkpointing
   EOF

   if [ ! -e $RM_OUTPUT ]; then
      deepspeed --module $get_rewards_commands
      checkSuccess "RM"
   fi
   deepspeed --module $sft_commands

Options for ``batch_inference``, refer to :ref:`batch_inference`.
Extra options for ``Conditional SFT``:

- ``--reward_template``: default value is ``'{input} <rm_score>: {reward} '``


Knowledge Distillation (MiniLLM)
------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_kd \
      --max_len 4096 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template 'User: {}\nAssistant: ' \
      --train_batch_size 256 \
      --micro_train_batch_size 2 \
      --max_samples 500000 \
      --pretrain meta-llama/Llama-2-7b-hf \
      --teacher_model meta-llama/Llama-2-13b-chat-hf \
      --save_path ./checkpoint/llama2-7b-kd \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --zero_stage 3 \
      --max_epochs 1 \
      --bf16 \
      --flash_attn \
      --kd_coef 0.4 \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: Input JSON Key for conversions
- ``--teacher_model``: Teacher model
- ``--teacher_offload``: Offload Teacher model to CPU
- ``--kd_coef``: KD Loss Coef, see `MiniLLM <https://github.com/microsoft/LMOps/tree/main/minillm>`_

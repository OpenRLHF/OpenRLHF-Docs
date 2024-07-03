Usage
=====

Common Options
---------------

We provide launch scripts for supported algorithms in the `examples <https://github.com/OpenLLMAI/OpenRLHF/tree/main/examples>`_ directory.
Here are the common options for the supported algorithms.

- ``--zero_stage``: DeepSpeed ZeRO Stage
- ``--max_len``: Max length for the datasets
- ``--adam_offload``: Offload the Adam Optimizer to GPU
- ``--bf16``: Enable bfloat16
- ``--flash_attn``: Enable Flash Attention 2
- ``--gradient_checkpointing``: Enable Gradient Checkpointing
- ``--use_wandb``: Set to ``{wandb_token}`` or ``True`` with shell command ``wandb login``
- ``--learning_rate``: Learning Rate
- ``--l2``: Weight Decay
- ``--lr_scheduler``: Learning Rate Scheduler 
- ``--max_samples``: Max training samples
- ``--max_norm``: Gradient clipping
- ``--micro_train_batch_size``: Batch size per GPU for training
- ``--train_batch_size``: Global training batch size
- ``--dataset``: Datasets Names or Paths
- ``--dataset_probs``: Datasets Mixing Probs
- ``--input_key``: Input JSON Key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--aux_loss_coef``: Balancing loss coef for MoE
- ``--load_in_4bit``: Use QLoRA
- ``--lora_rank``: Set to ``integer > 0`` to enable LoRA
- ``--lora_dropout``: LoRA dropout for HuggingFace PEFT (LoRA)
- ``--target_modules``: ``target_modules`` for HuggingFace PEFT (LoRA)
- ``--save_path``: final huggingface model save patch
- ``--max_epoch``: training epochs


Supervised Fine-tuning
----------------------

.. code-block:: bash

   deepspeed ./train_sft.py \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --dataset_probs 1.0 \
      --train_batch_size 256 \
      --micro_train_batch_size 2 \
      --max_samples 500000 \
      --pretrain meta-llama/Llama-2-7b-hf \
      --save_path ./checkpoint/llama2-7b-sft \
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

- ``--input_key``: JSON dataset Input Key for conversions
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--pretrain_mode``: Continue pretrain mode



Reward Model Training
---------------------

.. code-block:: bash

   deepspeed ./train_rm.py \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 9e-6 \
      --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku \
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
- ``--head_prefix``: custom ``value_head`` (score head) prefix


.. _rayppo:

PPO with Ray
------------

To improve RLHF training speed or support 70B models, we can use the ``PPO with Ray and vLLM acceleration``

.. code-block:: bash
   
   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # if you want to launch ray on more nodes, use
   ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt"}' \
      -- python3 examples/train_ppo_ray.py \
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
      --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
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
      --prompt_data OpenLLMAI/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --normalize_reward \
      --adam_offload \
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

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

- ``--save_value_network``: Save the Value Network after training is complete
- ``--normalize_reward``: Enable Reward Normalization
- ``--head_prefix``: custom ``value_head``  (score head) prefix for the reward model
- ``--init_kl_coef``: KL penalty coeff
- ``--max_epochs``: Number of PPO training epochs
- ``--micro_train_batch_size``: Batch size per GPU for generation
- ``--train_batch_size``: PPO mini-batch size
- ``--micro_rollout_batch_size``: Batch size per GPU for training
- ``--rollout_batch_size``: Replay Buffer Size
- ``--prompt_max_len``: Max length for the prompts
- ``--generate_max_len``: Max length for the responses
- ``--n_samples_per_prompt``: Generate n samples for each promot
- ``--freezing_actor_steps``: Freezing the actor parameters to init critic in the first n steps
- ``--reward_pretrain``: can be set to multiple reward models, such as ``RewardMode1,RewardModel2,RewardModel3``
- ``--actor_learning_rate``: actor model learning rate
- ``--critic_learning_rate``: critic model learning rate

.. note:: Ray + vLLM does not supports LoRA currently. vLLM 0.4.2 is recommended to deploy due to the compatibility of NCCL.


Direct Preference Optimization (DPO)
-----------------------------------

.. code-block:: bash

   deepspeed ./train_dpo.py \
      --save_path ./checkpoint/llama3-8b-dpo \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 9e-6 \
      --beta 0.1 \
      --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku\
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


Kahneman-Tversky Optimization (KTO)
------------------------------------

.. code-block:: bash

   deepspeed ./train_kto.py \
      --save_path ./checkpoint/llama3-8b-kto \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 9e-6 \
      --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku\
      --apply_chat_template \
      --chosen_key chosen \
      --rejected_key rejected \
      --flash_attn \
      --beta 0.1 \
      --gradient_checkpointing \
      --vanilla_loss \
      --use_wandb {wandb_token}

Options

- ``--chosen_key`` JSON dataset key for chosen conversions
- ``--rejected_key`` JSON dataset key for rejected conversions
- ``--vanilla_loss`` [for same num +/- samples in KTO batch]
- ``--ref_offload`` Offload Reference Model to CPU

support unpaired-preference dataset, like the following:

- ``--dataset {Datasets Name or Path}`` \
- ``--output_key {JSON dataset Key Name}`` \
- ``--unpaired_preference``


Rejection Sampling & RAFT
-------------------------

.. code-block:: bash

   checkSuccess() {
      if [[ $? != 0 ]]; then
         echo "FAILED $1"
         exit 1
      fi
   }

   mkdir -p ./checkpoint/llama-2-7b-rejection
   GENERATE_OUTPUT=./checkpoint/llama-2-7b-rejection/generate.jsonl
   RM_OUTPUT=./checkpoint/llama-2-7b-rejection/rm.jsonl
   ITER_LOG_PATH=./checkpoint/llama-2-7b-rejection/iter.log
   MODEL_OUTPUT_PATH=./checkpoint/llama-2-7b-rejection

   TRAINING_ITERS=20
   ROLLOUT_BATCH_SIZE=2048

   POLICY_MODEL_PATH=OpenLLMAI/Llama-2-7b-sft-model-ocra-500k

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
   ./batch_inference.py
      --eval_task generate_vllm \
      --pretrain $POLICY_MODEL_PATH \
      --bf16 \
      --max_len 2048 \
      --max_samples 128 \
      --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf  \
      --dataset_probs 0.5,0.5 \
      --temperature 0.9
      --zero_stage 0 \
      --best_of_n 4 \
      --enable_prefix_caching \
      --tp_size 4 \
      --micro_batch_size 64 \
      --iter $iter \
      --rollout_batch_size $ROLLOUT_BATCH_SIZE \
      --output_path $GENERATE_OUTPUT
   EOF
      echo $generate_commands
      python $generate_commands
      checkSuccess "GENERATE"

      read -r -d '' get_rewards_commands <<EOF
   ./batch_inference.py
      --eval_task rm \
      --pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
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
      deepspeed $get_rewards_commands
      checkSuccess "RM"

      read -r -d '' sft_commands <<EOF
   ./train_sft.py \
      --max_len 2048 \
      --dataset $RM_OUTPUT \
      --dataset_probs 1.0 \
      --train_batch_size 128 \
      --micro_train_batch_size 2 \
      --pretrain $POLICY_MODEL_PATH \
      --save_path ./checkpoint/llama-2-7b-rejection \
      --lr_scheduler constant \
      --zero_stage 2 \
      --max_epochs 1 \
      --bf16 \
      --learning_rate 2e-6 \
      --gradient_checkpointing
   EOF
      echo $sft_commands
      deepspeed $sft_commands
      checkSuccess "SFT"

      iter=$((iter + 1))
      if [[ "$ITER_LOG_PATH" != "null" ]]; then
         echo $iter >$ITER_LOG_PATH
      fi
   done

.. _batch_inference:

Options for ``batch_inference.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--eval_task``: set to ``generate_vllm``, ``generate`` (HF generate) or ``rm``
- ``--iter``: used to slice the datasets in range ``iter * rollout_batch_size: (iter + 1) * rollout_batch_size``
- ``--rollout_batch_size``: number of samples to generate
- ``--best_of_n``: number of responses to generate per prompt
- ``--input_key``: JSON dataset input key
- ``--tp_size``: TP Size for vLLM
- ``--enable_prefix_caching``: Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation
- ``--max_new_tokens``: Max new tokens in generation
- ``--greedy_sampling``: Use Greedy sampling
- ``--head_prefix``: ``value_head`` prefix for Reward Model
- ``--post_processor``: set to ``rs`` (Rejection Sampling), ``ca`` (Conditional SFT), ``iter_dpo`` (Iterative DPO) or None


Iterative DPO (RLHFlow)
------------

.. code-block:: bash

   checkSuccess() {
      if [[ $? != 0 ]]; then
         echo "FAILED $1"
         exit 1
      fi
   }

   mkdir -p ./checkpoint/llama-2-7b-iter-dpo
   GENERATE_OUTPUT=./checkpoint/llama-2-7b-iter-dpo/generate.jsonl
   RM_OUTPUT=./checkpoint/llama-2-7b-iter-dpo/rm.jsonl
   MODEL_OUTPUT_PATH=./checkpoint/llama-2-7b-iter-dpo/checkpoint
   ITER_LOG_PATH=null

   TRAINING_ITERS=5
   ROLLOUT_BATCH_SIZE=10240

   POLICY_MODEL_PATH=OpenLLMAI/Llama-2-7b-sft-model-ocra-500k
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
   ./batch_inference.py
      --eval_task generate_vllm \
      --pretrain $POLICY_MODEL_PATH \
      --max_new_tokens 1024 \
      --dataset Open-Orca/OpenOrca,Dahoas/full-hh-rlhf  \
      --dataset_probs 0.5,0.5 \
      --temperature 1.0 \
      --tp_size 4 \
      --best_of_n 16 \
      --enable_prefix_caching \
      --max_num_seqs 128 \
      --iter $iter \
      --rollout_batch_size $ROLLOUT_BATCH_SIZE \
      --output_path $GENERATE_OUTPUT
   EOF
      echo $generate_commands
      python $generate_commands
      checkSuccess "GENERATE"

      read -r -d '' get_rewards_commands <<EOF
   ./batch_inference.py
      --eval_task rm \
      --pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
      --bf16 \
      --max_len 2048 \
      --dataset $GENERATE_OUTPUT  \
      --dataset_probs 1.0 \
      --zero_stage 0 \
      --post_processor iter_dpo \
      --micro_batch_size 4 \
      --output_path $RM_OUTPUT
   EOF
      echo $get_rewards_commands
      deepspeed $get_rewards_commands
      checkSuccess "RM"

      read -r -d '' dpo_commands <<EOF
   ./train_dpo.py \
      --max_len 2048 \
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
      deepspeed $dpo_commands
      checkSuccess "DPO"

      iter=$((iter + 1))
      if [[ "$ITER_LOG_PATH" != "null" ]]; then
         echo $iter >$ITER_LOG_PATH
      fi
   done

Options for ``batch_inference.py``, refer to :ref:`batch_inference`.


Knowledge Distillation (MiniLLM)
------------

.. code-block:: bash

   deepSpeed ./train_kd.py \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --dataset_probs 1.0 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
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

Non-RL Methods
==============

This page covers supervised / preference-based / non-RL training methods (e.g., SFT, DPO, KTO) and iterative data filtering workflows.

.. _train_sft:

Supervised Fine-tuning (SFT)
----------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 2048 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\\nAssistant: ' \
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
      --attn_implementation flash_attention_2 \
      --packing_samples \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options
~~~~~~~

- ``--input_key``: JSON dataset key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--pretrain_mode``: Continue pretrain mode
- ``--packing_samples``: Packing SFT samples
- ``--multiturn``: Enable multi turn fine-tuning loss

.. note::
   OpenRLHF SFT/DPO/PPO/RM trainers support ``--packing_samples`` `using flash_attention <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_

Direct Preference Optimization (DPO)
------------------------------------

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
      --attn_implementation flash_attention_2 \
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
- ``--nll_loss_coef``: Regularization with NLL loss (See Llama 3.1 tech report)


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
      --input_template $'User: {}\nAssistant: ' \
      --attn_implementation flash_attention_2 \
      --beta 0.1 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: JSON dataset key for the instructions
- ``--output_key``: JSON dataset key for the responses
- ``--label_key``: JSON dataset key for the label
- ``--ref_offload``: Offload Reference Model to CPU
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template`` (Use ``--input_key`` to specify ``conversions``)

.. _train_prm:

Process Reward Model (PRM) Training
--------------------------------------------------

PRM training is a supervised workflow that learns **process-level reward signals** from labeled data. Although it is often used together with RL, it is not an RL update itself, so we keep the recipe here alongside other non-RL / data-flow methods.

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_prm \
      --save_path ./checkpoint/mistal-7b-prm \
      --save_steps 500 \
      --logging_steps 1 \
      --eval_steps 100 \
      --train_batch_size 256 \
      --micro_train_batch_size 8 \
      --pretrain mistralai/Mistral-7B-v0.1  \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 1e-6 \
      --dataset peiyi9979/Math-Shepherd \
      --input_key input \
      --label_key label \
      --attn_implementation flash_attention_2 \
      --load_checkpoint \
      --gradient_checkpointing \
      --packing_samples \
      --wandb_group prm \
      --placeholder_token "ки" \
      --reward_tokens "+" "-"

Options
^^^^^^^

- ``--input_key``: JSON dataset key for input text
- ``--label_key``: JSON dataset key for reward label
- ``--placeholder_token``: Step placeholder token
- ``--reward_tokens``: Reward label tokens


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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
-------------

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
      --prompt_key prompt \
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
---------------

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
--------------------------------------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_kd \
      --max_len 4096 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\nAssistant: ' \
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
      --attn_implementation flash_attention_2 \
      --kd_coef 0.4 \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: Input JSON Key for conversions
- ``--teacher_model``: Teacher model
- ``--teacher_offload``: Offload Teacher model to CPU
- ``--kd_coef``: KD Loss Coef, see `MiniLLM <https://github.com/microsoft/LMOps/tree/main/minillm>`_

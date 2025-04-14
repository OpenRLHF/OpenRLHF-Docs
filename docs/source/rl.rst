Reinforcement Learning from Human Feedback (RLHF)
=====

Common Options
---------------

We provide launch scripts for supported algorithms in the `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_ directory.
Here are some common options for the supported algorithms.

Training

- ``--zero_stage``: DeepSpeed ZeRO Stage
- ``--adam_offload``: Offload the Adam Optimizer to CPU
- ``--adam_betas``: Adam betas, default value is ``(0.9, 0.95)``
- ``--overlap_comm``: Enable backward & gradient overlap_comm for Deepspeed (overlap_comm uses 4.5x the allgather_bucket_size and reduce_bucket_size values.)
- ``--bf16``: Enable bfloat16
- ``--flash_attn``: Enable Flash Attention 2
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

Datasets

- ``--dataset``: Dataset names or paths for training
- ``--dataset_probs``: Dataset mixing probabilities
- ``--eval_dataset``: Dataset names or paths for evaluation
- ``--input_key``: Input JSON key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--input_template``: Custom ``input_template`` (when not using ``tokenizer.apply_chat_template``), set to ``None`` to disable it. Such as ``$'User: {}\nAssistant: '``.
- ``--max_len``: Max length for the samples
- ``--max_samples``: Max training samples
- ``--packing_samples``: Packing samples using Flash Attention 2

LoRA

- ``--load_in_4bit``: Use QLoRA
- ``--lora_rank``: Set to ``integer > 0`` to enable LoRA
- ``--lora_dropout``: LoRA dropout for HuggingFace PEFT (LoRA)
- ``--target_modules``: Target modules for HuggingFace PEFT (LoRA)

If you use ``LoRA (Low-Rank Adaptation)``, ``OpenRLHF`` will not save the full weights by default instead of ``LoRA Adapter``. To continue in your task normally, you should combine the ``Adapter`` with weights of your base model

.. code-block:: bash

   python -m openrlhf.cli.lora_combiner \
      --model_path meta-llama/Meta-Llama-3-8B \
      --lora_path ./checkpoint/llama3-8b-rm \
      --output_path ./checkpoint/llama-3-8b-rm-combined \
      --is_rm \
      --bf16


Supervised Fine-tuning
----------------------

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
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--input_key``: JSON dataset key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--pretrain_mode``: Continue pretrain mode
- ``--packing_samples``: Packing SFT samples
- ``--multiturn``: Enable multi turn fine-tuning loss

.. note:: OpenRLHF SFT/DPO/RM trainers support ``--packing_samples`` `using --flash_attn <https://github.com/MeetKai/functionary/tree/main/functionary/train/packing>`_



Reward Model Training
---------------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 4 \
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
      --packing_samples \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--chosen_key`` JSON dataset key for chosen conversions
- ``--rejected_key`` JSON dataset key for rejected conversions
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--value_head_prefix``: custom ``value_head`` (score head) prefix
- ``--packing_samples``: Packing RM samples

It is recommended to set the ``--value_prefix_head`` option of the Reward Model to ``score``, so that we can load the model using ``AutoModelForSequenceClassification``:

.. code-block:: python

   reward_model = AutoModelForSequenceClassification.from_pretrained(
               reward_model_path,
               num_labels=1,
               torch_dtype=torch.bfloat16,
               attn_implementation="flash_attention_2",
               use_cache=False,
            )
   inputs = xxxx (Left Padding Input Tokens)
   reward = reward_model.model(*inputs).last_hidden_state
   reward = reward_model.score(reward)[:, -1]


Process Reward Model (PRM) Training
---------------------

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
      --flash_attn \
      --load_checkpoint \
      --gradient_checkpointing \
      --packing_samples \
      --wandb_group prm \
      --placeholder_token "ки" \
      --reward_tokens "+" "-"

Options

- ``--input_key`` JSON dataset key for input text
- ``--label_key`` JSON dataset key for reward label
- ``--placeholder_token`` step placeholder token
- ``--reward_tokens`` reward label


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
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
      --micro_train_batch_size 16 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 32 \
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
      --packing_samples \
      --normalize_reward \
      --adam_offload \
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}


.. note:: It is recommended to use a hybrid engine :ref:`hybrid_engine` to avoid resource idling.
.. note:: Do not set ``--vllm_num_engines`` means not using the vLLM engine. Ray + vLLM does not supports LoRA currently. You can also use ``setup_commands`` to let Ray automatically deploy the environment, such as ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'``
.. note:: If you want to run on AMD GPUs, or for whatever reason you encounter an error related to index out of range when deepspeed sets up the GPU devices, you can try to set the environment variable `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/utils.py>`_ as a workaround.
.. code-block:: bash

   # For NVIDIA GPUs:
   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
   # For AMD GPUs:
   export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

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
- ``--enforce_eager``: Disable cuda graph for vLLM
- ``--ref_reward_offload``: Offload Reward and Reference models to CPU when enabling Hybrid Engine 
- ``--vllm_sync_backend``: Set to ``nccl`` or ``gloo`` for vLLM weights sync. We recommend using vLLM 0.8.3+, as other versions currently require synchronizing weights via Gloo (``--vllm_sync_backend gloo``). 
- ``--vllm_sync_with_ray``: Use `ray.util.collective <https://docs.ray.io/en/latest/ray-more-libs/ray-collective.html>`_ to synchronize vLLM weights and avoid NCCL hang.
- ``--enable_prefix_caching``: Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation
- ``--packing_samples``: Packing PPO samples in training and forward

PPO

- ``--save_value_network``: Save the Value Network after training is complete
- ``--normalize_reward``: Enable Reward Normalization
- ``--value_head_prefix``: custom ``value_head``  (score head) prefix for the reward model
- ``--init_kl_coef``: KL penalty coeff
- ``--max_epochs``: Number of PPO training epochs in a global step
- ``--num_episodes``: Number of PPO episodes (the number of data passes)
- ``--micro_train_batch_size``: Batch size per GPU for training
- ``--train_batch_size``: PPO mini-batch size
- ``--micro_rollout_batch_size``: Batch size per GPU for generation
- ``--rollout_batch_size``: Replay Buffer Size ``= rollout_batch_size * n_samples_per_prompt``
- ``--prompt_max_len``: Max length for the prompts
- ``--generate_max_len``: Max length for the responses
- ``--n_samples_per_prompt``: Generate n samples for each promot
- ``--eval_n_samples_per_prompt``: Number of samples for evaluation
- ``--freezing_actor_steps``: Freezing the actor parameters to init critic in the first n steps
- ``--reward_pretrain``: Can be set to a reward model which is used for reward and critic initialization
- ``--actor_learning_rate``: Actor model learning rate
- ``--critic_learning_rate``: Critic model learning rate
- ``--reward_clip_range``: Reward value cliprange, such as ``(-10, 10)``
- ``--temperature``: PPO samling temperature for LLMs
- ``--eval_temperature``: PPO samling temperature for evaluation
- ``--gamma``: ``gamma`` for RL, default value is ``1.0``
- ``--lambd``: ``lambda`` for GAE, default value is ``1.0``
- ``--no_advantage_std_norm``: disable dividing by std for advantages while keeping mean normalization

Datasets

- ``--prompt_data``: Dataset names or paths (Prompts)
- ``--prompt_data_probs``: Dataset mixing probabilities
- ``--pretrain_data``: Dataset names or paths (Pretrain)
- ``--pretrain_data_probs``: Dataset mixing probabilities
- ``--eval_dataset``: Dataset names or paths for evaluation


REINFORCE++ /RLOO with Ray (vLLM)
------------

In REINFORCE-like algorithms, the value network is not used; instead, advantage is calculated directly by normalizing the reward, which can save some computational resources.
We also proposed the `REINFORCE++ <https://www.researchgate.net/publication/387487679_REINFORCE_A_SIMPLE_AND_EFFICIENT_APPROACH_FOR_ALIGNING_LARGE_LANGUAGE_MODELS>`_ alignment method.

- REINFORCE++ incorporates ``key optimization techniques from PPO`` while completely eliminating the need for a critic network.
- REINFORCE++-baseline leverages the ``mean reward across multiple samples generated from the same prompt`` as a baseline for reward reshaping (with global batch normalization ``/std``).
- RLOO implementation in OpenRLHF enhances the original algorithm by introducing per-token KL reward and adopting the PPO-clip loss mechanism.
- GRPO functionality can be activated by configuring ``--advantage_estimator group_norm`` along with K3 KL loss.
- Dr. GRPO represents a variant that eliminates the ``/std`` normalization present in GRPO.

.. code-block:: bash
   
   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # if you want to launch ray on more nodes, use
   ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 1 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 1 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
      --micro_train_batch_size 16 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 32 \
      --rollout_batch_size 128 \
      --n_samples_per_prompt 1 \
      --max_samples 100000 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --generate_max_len 1024 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --init_kl_coef 0.01 \
      --advantage_estimator reinforce \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --packing_samples \
      --normalize_reward \
      --adam_offload \
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options

- ``--advantage_estimator`` set to ``gae`` (for PPO), ``reinforce``, ``rloo``, ``reinforce_baseline`` or ``group_norm`` (for GRPO) or ``dr_grpo`` (for DR-GRPO)
- ``--use_kl_loss`` Add KL loss (required for GRPO) into policy loss and disable KL reward


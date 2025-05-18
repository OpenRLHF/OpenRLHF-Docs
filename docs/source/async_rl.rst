Asynchronous RL and Agent RL
=====


Asynchronous Agent RL
------------

.. _async_rl:

OpenRLHF provides comprehensive support for both Asynchronous RLHF and Agent-based RLHF implementations. To utilize these features, simply include the ``--async_train`` and ``--agent_func_path`` parameters in your training configuration. 

.. code-block:: bash
   
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 6 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 6 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 1 \
      --colocate_all_models \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /openrlhf/examples/python/agent_func.py \
      --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
      --micro_train_batch_size 16 \
      --train_batch_size 192 \
      --micro_rollout_batch_size 32 \
      --rollout_batch_size 192 \
      --n_samples_per_prompt 8 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --max_samples 12500 \
      --generate_max_len 1024 \
      --advantage_estimator reinforce_baseline \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 1e-6 \
      --init_kl_coef 1e-3 \
      --use_kl_loss \
      --kl_estimator k2 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --normalize_reward \
      --adam_offload \
      --gradient_checkpointing \
      --packing_samples \
      --async_train

``--agent_func_path`` indicates the path to the agent function, such as:

.. code-block:: python
   # agent_func.py
   step_idx = 0
   max_steps = 2

   async def step(state, action, label, **kwargs) -> Tuple[float, Dict[str, Any], bool]:
      global step_idx, max_steps
      # End after verification
      if step_idx >= max_steps:
         done = True
         # Generate a random reward using torch.rand
         reward = torch.rand(1)
         next_state = state + action + " The answer is correct. <|endoftext|>"
      else:
         done = False
         reward = torch.tensor(0)
         # Update state
         next_state = state + action + " The answer is not correct, please try again: "
      step_idx += 1

      # Extra info
      extra_info = {}
      return reward, next_state, done, extra_info

You can also configure the maximum number of concurrent agents per vLLM engine by setting ``export OPENRLHF_ASYNC_NUM_TASKS=128``. 
Additionally, you can control the degree of off-policy sampling by setting ``export OPENRLHF_ASYNC_QUEUE_SIZE=1`` (this parameter controls how many batches of data can be stored in the buffer at most) in your environment.



Synchronous Agent RL using Hybrid Engine
------------

Asynchronous training may affect the training stability. It is recommended to prioritize using Hybrid Engine or synchronous training mode.

.. code-block:: bash

   export PYTORCH_NVML_BASED_CUDA_CHECK=1
   export VLLM_USE_V1=1

   python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 8 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 8 \
      --vllm_num_engines 4 \
      --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.6 \
      --init_kl_coef 1e-3 \
      --use_kl_loss \
      --kl_estimator k3 \
      --advantage_estimator group_norm \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /openrlhf/examples/python/agent.py \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 8 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 16 \
      --rollout_batch_size 128 \
      --n_samples_per_prompt 8 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --max_samples 100000 \
      --generate_max_len 1024 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --gradient_checkpointing \
      --packing_samples \
      --vllm_sync_backend nccl \
      --enforce_eager \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep
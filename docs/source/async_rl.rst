Asynchronous RL and Agent RL
=====

In Agent RL training, the agent environments, vLLM, and DeepSpeed typically operate asynchronously, similar to traditional game RL. Therefore, OpenRLHF supports an asynchronous execution mode between these three components to better facilitate Agent RL training.


Asynchronous Agent RL
------------

.. _async_rl:

OpenRLHF provides comprehensive support for both Asynchronous RLHF and Agent-based RLHF implementations. To utilize these features, simply include the ``--async_train`` and ``--agent_func_path`` parameters in your training configuration. 

.. code-block:: bash
   # Required for Async LLM + Hybrid Engine
   export VLLM_USE_V1=1
   
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

Note that ``--colocate_all_models`` with ``--async_train`` only merge the deepspeed models, not the vllm engines.
The ``--agent_func_path`` indicates the path to the agent function, such as:

.. code-block:: python

   import random
   from typing import Any, Dict

   import torch

   # Global states for the environment
   step_idx = 0
   max_steps = random.randint(0, 2)

   # A n-step random environment
   async def step(observation, action, label, **kwargs) -> Dict[str, Any]:
      """Execute one step of verification and return a random reward using torch.rand

      Args:
         observation: The input prompt/expression
         action: The language model's response
         label: Agent identifier or additional information

      Returns:
         Dict[str, Any]: A dictionary containing:
               - rewards: Reward value for advantage calculation
               - scores: Reward value for dynamic filtering
               - next_observation: The updated observation after the step
               - done: Boolean indicating if the episode is complete
               - sampling_params: Parameters for vLLM sampling
               - extra_logs: Additional logging information
      """
      global step_idx, max_steps
      print(f"step_idx: {step_idx}, max_steps: {max_steps}")

      # End after verification
      if step_idx >= max_steps:
         done = True
         # Generate a random reward using torch.rand
         reward = torch.randint(0, 2, (1,)).float()
         next_observation = (
               observation
               + action
               + "\n\nHuman: [VERIFICATION RESULT: CORRECT]\nYour solution is valid and complete. The verification process is finished.\n</s>"
         )
      else:
         done = False
         reward = torch.tensor(0)
         # Update observation
         next_observation = (
               observation
               + action
               + "\n\nHuman: [VERIFICATION RESULT: INCORRECT]\nLet's analyze what needs improvement:\n1. What are the key issues in the current solution?\n2. How can we make it more robust?\n3. What additional considerations should we take into account?\n\nPlease provide your revised solution:\n</s>\n\nAssistant: "
         )
      step_idx += 1

      return {
         "rewards": reward,  # Rewards for advantage calculation
         "scores": reward,  # Scores for dynamic filtering (0-1 reward)
         "next_observation": next_observation,  # The updated observation for vLLM inference in next step
         "done": done,  # Boolean indicating if the episode is complete
         "sampling_params": kwargs.get("sampling_params", None),  # Parameters for vLLM sampling in next step
         "extra_logs": {"dummy_scores": reward},  # Additional logging information
      }


You can also configure the maximum number of concurrent agents per vLLM engine by setting ``export OPENRLHF_ASYNC_NUM_TASKS=128``. 
Additionally, you can control the degree of off-policy sampling by setting ``export OPENRLHF_ASYNC_QUEUE_SIZE=1`` (this parameter controls how many batches of data can be stored in the buffer at most) in your environment.



Synchronous Agent RL using Hybrid Engine
------------

Asynchronous training may affect the training stability. It is recommended to prioritize using Hybrid Engine or synchronous training mode.

.. code-block:: bash
   
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
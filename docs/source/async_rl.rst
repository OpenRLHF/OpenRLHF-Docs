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

The Agent API has been redesigned to use a class-based approach with ``AgentInstanceBase`` and ``AgentExecutorBase`` classes for better modularity and extensibility.

.. code-block:: python

   import random
   from typing import Any, Dict

   import torch
   from openrlhf.utils.agent import AgentExecutorBase, AgentInstanceBase


   # A simple n-step random environment
   class AgentInstance(AgentInstanceBase):
      async def __init__(self, *args, **kwargs):
         self.step_idx = 0
         self.max_steps = random.randint(1, 3)  # 1-3 steps

      async def reset(self, states: dict, **kwargs):
         """Initialize the environment and return initial observation

         Args:
               states: Dictionary containing prompt and label

         Returns:
               str: Initial observation text
         """
         return {"observation": states["observation"]}  # Return original text observation

      async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
         """Execute one step of verification and return environment feedback

         Args:
               states: Dictionary containing observation_text, action_text, and label

         Returns:
               Dict[str, Any]: A dictionary containing:
                  - rewards: Reward value for advantage calculation
                  - scores: Reward value for dynamic filtering
                  - environment_feedback: The environment feedback text
                  - done: Boolean indicating if the episode is complete
                  - sampling_params: Parameters for vLLM sampling
                  - extra_logs: Additional logging information
         """
         print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

         observation_text = states["observation_text"]
         action_text = states["action_text"]
         label = states["label"]

         # Check if episode is done
         done = self.step_idx >= self.max_steps
         reward = torch.randint(0, 2, (1,)).float() if done else torch.tensor(0)

         # Generate environment feedback based on whether episode is done
         environment_feedback = (
               "\n\nHuman: [CORRECT]\n</s>"
               if done
               else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
         )

         self.step_idx += 1

         return {
               "rewards": reward,  # Rewards for advantage calculation
               "scores": reward,  # Scores for dynamic filtering (0-1 reward)
               "environment_feedback": environment_feedback,  # Environment feedback text
               "done": done,  # Boolean indicating if the episode is complete
               "sampling_params": states.get("sampling_params", None),  # Parameters for vLLM sampling in next step
               "extra_logs": {"dummy_scores": reward},  # Additional logging information
         }


   class AgentExecutor(AgentExecutorBase):
      def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
         super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

      async def execute(self, prompt, label, sampling_params):
         # You could override the execute function of AgentExecutorBase to add custom agent running logic
         return await super().execute(prompt, label, sampling_params)



You can also configure the maximum number of concurrent agents per vLLM engine by setting ``export OPENRLHF_ASYNC_NUM_TASKS=128``. 
Additionally, you can control the degree of off-policy sampling by setting ``export OPENRLHF_ASYNC_QUEUE_SIZE=1`` (this parameter controls how many batches of data can be stored in the buffer at most) in your environment.

.. note:: By overriding the ``execute`` function of ``AgentExecutorBase``, you can implement completely custom agent running processes. The design follows the **token-in-token-out principle** to ensure consistency between sampling and training samples, avoiding potential mismatches that could occur with text-level processing.

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
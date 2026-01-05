Multi-Turn Agent: Complex Environment Interactions
===================================================

For tasks requiring **multi-step interactions** (reasoning chains, coding with feedback, game playing), OpenRLHF provides the **Multi-Turn Agent Execution** mode.

This mode enables:

- Multi-step interactions with environment feedback
- Custom agent implementation via ``AgentInstanceBase``
- External environment integration (e.g., NeMo Gym)

See :doc:`agent_paradigm` for architecture overview.

Overview
--------

In Multi-Turn Agent mode, the agent:

1. Receives initial observation (prompt)
2. Generates action (response)
3. Gets environment feedback
4. Continues generation or terminates based on ``done`` flag
5. Receives reward when episode completes

This design follows the **token-in-token-out principle** to ensure perfect consistency between generation and training.

Building Custom Multi-Turn Agents
----------------------------------

Implement ``AgentInstanceBase`` with ``reset`` and ``step`` methods:

.. code-block:: python

    # agent_func.py
    import random
    from typing import Any, Dict

    import torch
    from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


    # A simple n-step random environment
    class AgentInstance(AgentInstanceBase):
        async def __init__(self, *args, **kwargs):
            self.step_idx = 0
            self.max_steps = random.randint(1, 3)  # 1-3 steps

        async def reset(self, states: dict, **kwargs):
            """Initialize the environment and return initial observation
            
            Args:
                states: Dictionary containing observation (prompt) and label
            
            Returns:
                dict: {"observation": str} - Initial observation text
            """
            return {"observation": states["observation"]}  # Return original text observation

        async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
            """Execute one step of verification and return environment feedback
            
            Args:
                states: Dictionary containing:
                    - observation_text: Current observation
                    - action_text: Model's generated action
                    - label: Ground truth label
            
            Returns:
                Dict[str, Any]: A dictionary containing:
                    - rewards: Reward value for advantage calculation
                    - scores: Reward value for dynamic filtering (0-1 range)
                    - environment_feedback: The environment feedback text
                    - done: Boolean indicating if the episode is complete
                    - sampling_params: Parameters for vLLM sampling in next step
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


    class AgentExecutor(MultiTurnAgentExecutor):
        def __init__(self):
            super().__init__(AgentInstance)

Agent API Components
--------------------

AgentInstanceBase Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

``async def reset(self, states: dict, **kwargs) -> dict``

- Called at the start of each episode
- **Args**: ``states`` contains ``observation`` (prompt) and ``label``
- **Returns**: ``{"observation": str}`` - Initial observation for the agent

``async def step(self, states: dict, **kwargs) -> Dict[str, Any]``

- Called after each model generation
- **Args**: ``states`` contains:
  
  - ``observation_text``: Current observation
  - ``action_text``: Model's generated response
  - ``label``: Ground truth label (from ``--label_key``)

- **Returns**: Dictionary with:
  
  - ``rewards``: Tensor for advantage calculation
  - ``scores``: Tensor for dynamic filtering (0-1 range)
  - ``environment_feedback``: Text feedback to append to context
  - ``done``: Boolean indicating if episode is complete
  - ``sampling_params``: Optional vLLM sampling parameters for next step
  - ``extra_logs``: Dictionary of metrics to log to wandb

AgentExecutor Class
~~~~~~~~~~~~~~~~~~~

Inherit from ``MultiTurnAgentExecutor`` and pass your ``AgentInstanceBase`` class:

.. code-block:: python

    class AgentExecutor(MultiTurnAgentExecutor):
        def __init__(self):
            super().__init__(AgentInstance)  # Pass your AgentInstance class

For fully custom token-level execution, inherit ``AgentExecutorBase`` and override the ``execute()`` method.

Training with Multi-Turn Agents
--------------------------------

Asynchronous Training (Higher Throughput)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For maximum throughput, use asynchronous training with ``--async_train``:

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

.. note:: 
   When using ``--colocate_all_models`` with ``--async_train``, only the DeepSpeed models are merged, not the vLLM engines.

Synchronous Training with Hybrid Engine (Better Stability)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better training stability, use synchronous training with Hybrid Engine:

.. code-block:: bash
   
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
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

Configuration Options
---------------------

Agent Configuration

- ``--agent_func_path``: Path to your agent implementation file

Async Training Options

- ``--async_train``: Enable asynchronous training
- ``export OPENRLHF_ASYNC_NUM_TASKS=128``: Max concurrent agents per vLLM engine
- ``export OPENRLHF_ASYNC_QUEUE_SIZE=1``: Buffer size (larger = more off-policy)

Hybrid Engine Options

- ``--colocate_all_models``: Share GPUs between vLLM and DeepSpeed models
- ``--vllm_gpu_memory_utilization 0.6``: GPU memory fraction for vLLM
- ``--vllm_enable_sleep``: Enable sleep mode for vLLM
- ``--deepspeed_enable_sleep``: Enable sleep mode for DeepSpeed

See :doc:`hybrid_engine` for more details on Hybrid Engine.

Training Modes Comparison
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Mode
     - Pros
     - Cons
   * - **Hybrid Engine (Sync)**
     - Best GPU utilization, stable training
     - Slightly lower throughput than async
   * - **Asynchronous**
     - Highest throughput
     - May affect training stability (off-policy)
   * - **Distributed (Sync)**
     - Simple, stable
     - GPU idle time between phases

.. tip::
   **Recommended Priority**: Hybrid Engine > Synchronous > Asynchronous
   
   Use Hybrid Engine for best balance of efficiency and stability. Only use asynchronous mode when throughput is critical and convergence is validated.

.. warning::
   Asynchronous training may affect training stability due to off-policy data. Monitor convergence carefully and consider reducing ``OPENRLHF_ASYNC_QUEUE_SIZE`` if needed.

Advanced: Custom Token-Level Execution
---------------------------------------

For complete control over the agent execution pipeline, inherit ``AgentExecutorBase`` and override the ``execute()`` method:

.. code-block:: python

    from openrlhf.utils.agent import AgentExecutorBase

    class CustomAgentExecutor(AgentExecutorBase):
        def __init__(self, max_steps, max_length, llm_engine, hf_tokenizer, result_queue):
            super().__init__(AgentInstance, max_steps, max_length, llm_engine, hf_tokenizer, result_queue)

        async def execute(self, prompt, label, sampling_params):
            """
            Fully custom token-level execution logic.
            
            Must follow the token-in-token-out principle to ensure
            consistency between sampling and training samples.
            """
            # Your custom execution logic here
            return await super().execute(prompt, label, sampling_params)

.. note:: 
   By overriding the ``execute`` function of ``AgentExecutorBase``, you can implement completely custom agent running processes. The design follows the **token-in-token-out principle** to ensure consistency between sampling and training samples, avoiding potential mismatches that could occur with text-level processing.

External Environment Integration
---------------------------------

NeMo Gym Integration
~~~~~~~~~~~~~~~~~~~~

OpenRLHF supports integration with `NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ for advanced agent-based RLHF training with external evaluation environments.

Example configuration:

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --agent_func_path /path/to/nemogym_agent.py \
      --pretrain your_model \
      ... # other training args

See the `NeMo Gym documentation <https://github.com/NVIDIA-NeMo/Gym>`_ and `OpenRLHF NeMo Gym example <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_nemogym.sh>`_ for integration details.

Algorithm Compatibility
-----------------------

All multi-turn agent implementations work with **any RL algorithm**. Switch algorithms via ``--advantage_estimator``. See :doc:`agent_paradigm` for available algorithms and :doc:`rl` for detailed usage.

Example Use Cases
-----------------

Multi-Step Math Reasoning
~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent receives feedback after each reasoning step, allowing it to correct mistakes and try alternative approaches.

Coding with Execution Feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent generates code, receives execution results and error messages, then iterates to fix bugs.

Game Playing
~~~~~~~~~~~~

Agent interacts with game environment, receives rewards and state updates, learns optimal strategies.

Interactive Question Answering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Agent can ask clarifying questions and receive answers before providing final response.


Agent-based Training Guide
==========================

OpenRLHF uses a unified **agent execution pipeline** for all training. This page consolidates:

- Training recipes (SFT / RM / RL with Ray + vLLM)
- Agent execution modes (single-turn, multi-turn)
- How to customize rewards and environments

For the underlying design principles, see :doc:`agent_paradigm`.

.. contents::
   :local:
   :depth: 2

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

.. _train_rm:

Reward Model (RM) Training
--------------------------

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
      --attn_implementation flash_attention_2 \
      --packing_samples \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

Options
~~~~~~~

- ``--chosen_key``: JSON dataset key for chosen conversions
- ``--rejected_key``: JSON dataset key for rejected conversions
- ``--tokenizer_chat_template``: Custom ``chat_template`` for HuggingFace tokenizer template
- ``--value_head_prefix``: Custom ``value_head`` (score head) prefix
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

.. _rayppo:

RL Training with Ray + vLLM (Agent Execution)
---------------------------------------------

All RL training in OpenRLHF uses the **agent execution pipeline** (single-turn by default). For execution-mode details, see :ref:`single_turn_mode` and :ref:`multi_turn_mode`.

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
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

.. note::
   The full Hybrid Engine (colocation) recipe is documented in :doc:`hybrid_engine`. Use that page as the canonical reference for ``--colocate_all_models``.

.. note::
   - Ray + vLLM does not support LoRA currently
   - Use ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`` to let Ray automatically deploy the environment
   - For AMD GPUs or GPU device index errors, set ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` (NVIDIA) or ``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`` (AMD)

RL Algorithms (PPO, REINFORCE++, GRPO, RLOO)
--------------------------------------------

**Key Design Principle**: RL algorithms are **completely decoupled** from agent execution modes.

- **Algorithm Selection**: Use ``--advantage_estimator`` flag
- **Execution Mode**: Default (single-turn) or multi-turn via ``--agent_func_path``
- **Independence**: Any algorithm works with any execution mode

Supported Algorithms
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Algorithm
     - ``--advantage_estimator``
     - Key Feature
     - Best Use Case
   * - **PPO**
     - ``gae`` (default)
     - Full critic network
     - Stable training, proven results
   * - **REINFORCE++**
     - ``reinforce``
     - PPO tricks without critic
     - Efficient training, less memory
   * - **REINFORCE++-baseline**
     - ``reinforce_baseline``
     - Mean reward baseline
     - Reasoning tasks (RLVR), robust to reward scales
   * - **RLOO**
     - ``rloo``
     - Per-token KL + PPO-clip
     - Multi-sample training
   * - **GRPO**
     - ``group_norm``
     - Group normalization
     - Batch-based training
   * - **Dr. GRPO**
     - ``dr_grpo``
     - Simplified GRPO
     - Removes local /std norm

.. _single_turn_mode:

Single-turn mode (default): Remote RM & Custom Rewards
------------------------------------------------------

The **single-turn agent execution** (default) covers most RLHF use cases.

Remote Reward Model Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we have deployed a large reward model (Llama3-405B, ArmoRM-Llama3-8B-v0.1, or PairRM) on a remote server. OpenRLHF provides an HTTP interface to use these models in training.

Starting the Remote Reward Model Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`openrlhf.cli.serve_rm <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/serve_rm.py#L1>`_

Launch the reward model server:

.. code-block:: bash

   python -m openrlhf.cli.serve_rm \
       --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
       --port 5000 \
       --bf16 \
       --attn_implementation flash_attention_2 \
       --normalize_reward \
       --max_len 8192 \
       --batch_size 16

See also the upstream example scripts directory: `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

Using Remote Reward Model in Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Then, specify ``--remote_rm_url`` during RL training:

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json='{"working_dir": "/openrlhf"}' \
       -- python3 -m openrlhf.cli.train_ppo_ray \
       --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
       --remote_rm_url http://localhost:5000/get_reward \
       --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
       ... # other training args

Reinforced Fine-tuning with Custom Reward Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also pass a local Python reward function file to ``--remote_rm_url``.

.. _multi_turn_mode:

Multi-turn mode: environments & async training
----------------------------------------------

For tasks requiring **multi-step interactions** (reasoning chains, coding with feedback, game playing), use multi-turn agent execution via ``--agent_func_path``.

Execution loop (mental model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In multi-turn mode, OpenRLHF treats each sample as an *episode*:

1. Reset environment with the initial prompt/label
2. Model generates an action (text)
3. Environment returns feedback + optional reward and whether the episode is done
4. Repeat until ``done=True``

The design follows the **token-in-token-out** principle to keep generation and training aligned.

Minimal agent implementation (agent_func.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You provide a Python file via ``--agent_func_path``. It must implement an ``AgentInstanceBase`` and optionally an ``AgentExecutor`` wrapper.

.. code-block:: python

   # agent_func.py (minimal example)
   from typing import Any, Dict
   import torch

   from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


   class AgentInstance(AgentInstanceBase):
       def __init__(self, *args, **kwargs):
           self.step_idx = 0

       async def reset(self, states: dict, **kwargs) -> dict:
           # states typically contains: observation (prompt) and label (ground truth)
           self.step_idx = 0
           return {"observation": states["observation"]}

       async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
           # states typically contains:
           # - observation_text: current context
           # - action_text: model output for this step
           # - label: ground truth label (from --label_key)
           self.step_idx += 1
           done = self.step_idx >= 3

           # Reward usually only provided at terminal step
           reward = torch.tensor(1.0) if done else torch.tensor(0.0)

           # Feedback appended to the next context
           environment_feedback = (
               "\n\nHuman: [CORRECT]\n</s>"
               if done
               else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
           )

           return {
               "rewards": reward,
               "scores": reward,
               "environment_feedback": environment_feedback,
               "done": done,
               "sampling_params": states.get("sampling_params", None),
               "extra_logs": {"step": self.step_idx},
           }


   class AgentExecutor(MultiTurnAgentExecutor):
       def __init__(self):
           super().__init__(AgentInstance)

Required methods / return schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``async def reset(self, states: dict, **kwargs) -> dict``

  - **Returns**: ``{"observation": str}``

- ``async def step(self, states: dict, **kwargs) -> Dict[str, Any]``

  - **Returns** a dict containing at least:

    - ``rewards``: Tensor (used for advantage calculation)
    - ``scores``: Tensor (used for dynamic filtering; often same as rewards)
    - ``environment_feedback``: str (appended to the context for the next step)
    - ``done``: bool

  - Optional:

    - ``sampling_params``: per-step vLLM sampling params
    - ``extra_logs``: logged metrics

Training with multi-turn agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synchronous (more stable):

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /path/to/agent_func.py \
      ... # other training args

Asynchronous (higher throughput, may affect stability):

.. code-block:: bash

   export VLLM_USE_V1=1
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /path/to/agent_func.py \
      --async_train \
      ... # other training args

Key knobs
~~~~~~~~~

- ``--agent_func_path``: path to your multi-turn agent file
- ``--async_train``: enable async training
- ``--async_queue_size``: async buffer size (larger may be more off-policy, default 1)

.. note::
   Async training is **mutually exclusive** with Hybrid Engine (do not combine ``--async_train`` with ``--colocate_all_models``). See :doc:`hybrid_engine`.

External environments
~~~~~~~~~~~~~~~~~~~~~

Multi-turn agents can wrap external evaluators/environments (e.g., game envs, execution-based code judges). For an end-to-end reference, see the upstream OpenRLHF examples and scripts.

Useful upstream examples
~~~~~~~~~~~~~~~~~~~~~~~~

- Async pipeline RLHF (``--async_train``): `train_reinforce_baseline_llama_ray_async.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_llama_ray_async.sh>`_
- Async agent RLHF (``--agent_func_path`` + async): `train_reinforce_baseline_llama_ray_agent_async.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_llama_ray_agent_async.sh>`_
- NeMo Gym integration: `train_reinforce_nemogym.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_nemogym.sh>`_

Common Options (Shared Across Trainers)
---------------------------------------

We provide launch scripts for supported algorithms in the `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_ directory.

Training
~~~~~~~~

- ``--zero_stage``: DeepSpeed ZeRO Stage
- ``--adam_offload``: Offload the Adam Optimizer to CPU
- ``--adam_betas``: Adam betas, default value is ``(0.9, 0.95)``
- ``--overlap_comm``: Enable backward & gradient overlap_comm for Deepspeed (overlap_comm uses 4.5x the allgather_bucket_size and reduce_bucket_size values.)
- ``--bf16``: Enable bfloat16
- ``--attn_implementation``: Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)
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
- ``--ds_tensor_parallel_size``: DeepSpeed Tensor Parallel Size (AutoTP), only used when ``--zero_stage 0 / 1 / 2``

Datasets
~~~~~~~~

- ``--dataset``: Dataset names or paths for training
- ``--dataset_probs``: Dataset mixing probabilities
- ``--eval_dataset``: Dataset names or paths for evaluation
- ``--input_key``: Input JSON key for conversions
- ``--apply_chat_template``: Use HuggingFace ``tokenizer.apply_chat_template``
- ``--input_template``: Custom ``input_template`` (when not using ``tokenizer.apply_chat_template``), set to ``None`` to disable it. Such as ``$'User: {}\\nAssistant: '``.
- ``--max_len``: Max length for the samples
- ``--max_samples``: Max training samples
- ``--packing_samples``: Packing samples using Flash Attention 2

LoRA
~~~~

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


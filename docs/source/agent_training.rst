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
      --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 8 \
      --critic_num_nodes 1 \
      --critic_num_gpus_per_node 8 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 8 \
      --vllm_num_engines 4 \
      --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.6 \
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
      --use_dynamic_batch \
      --normalize_reward \
      --adam_offload \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}
      --vllm_sync_backend nccl \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep

.. note::
   This example uses Hybrid Engine (``--colocate_all_models``). For detailed guidance and troubleshooting, see :doc:`hybrid_engine`.

.. note::
   - Ray + vLLM does not support LoRA currently
   - Use ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`` to let Ray automatically deploy the environment
   - For AMD GPUs or GPU device index errors, set ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` (NVIDIA) or ``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`` (AMD)

Dynamic sampling (multi-sample + dynamic filtering)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasoning tasks, it is common to generate **multiple rollouts per prompt** and then train only on a subset of them. OpenRLHF provides a *dynamic sampling* workflow via:

- ``--n_samples_per_prompt``: how many completions to generate for each prompt (multi-sample rollouts)
- ``--dynamic_filtering`` + ``--dynamic_filtering_reward_range <low> <high>``: filter samples by reward range (drop out-of-range samples)
- ``--use_dynamic_batch`` with token budgets: reduce padding waste for variable/long sequences

Guidelines:

- **Batch size relation**: a common choice is ``train_batch_size = rollout_batch_size * n_samples_per_prompt`` (e.g., ``128 * 8 = 1024``).
  With ``--dynamic_filtering``, the effective batch may become smaller if many samples are filtered out.
- **Reward range**: ``--dynamic_filtering_reward_range`` is applied to the scalar reward returned by your reward function / RM.
  If your rewards are not naturally in ``[0, 1]``, adjust the range (or enable ``--normalize_reward`` if appropriate).
- **Token budgets**: when using ``--use_dynamic_batch``, tune ``--train_max_tokens_per_gpu`` and ``--rollout_max_tokens_per_gpu`` to match your GPU memory and sequence lengths.

Off-policy correction: TIS and ICEPOP
-------------------------------------

When rollouts are generated by vLLM, OpenRLHF can apply **importance-sampling (IS) correction** to reduce mismatch between rollout log-probabilities and the training-time log-probabilities.

TIS (Truncated Importance Sampling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable vLLM IS correction and set the truncation interval:

- ``--enable_vllm_is_correction``
- ``--vllm_is_truncated_threshold <low> <high>`` (e.g., ``0.5 5.0``)

This truncates the vLLM IS ratio within a threshold interval for stability (see the off-policy note: `off-policy RL training <https://fengyao.notion.site/off-policy-rl>`_).

ICEPOP
~~~~~~

ICEPOP is an alternative to truncation: it **zeros out** IS coefficients outside the threshold interval instead of clipping.

Enable it with:

- ``--use_icepop``

Conceptually, it applies a mask (pseudo-code):

.. code-block:: python

   # ICEPOP: set coefficients outside the interval to 0
   vllm_is = exp(old_log_probs - rollout_log_probs)
   mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
   vllm_is = vllm_is * mask

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

- Async agent RLHF (``--agent_func_path`` + ``--async_train``): `train_reinforce_baseline_ray_agent_async.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_
- REINFORCE++-baseline + Hybrid Engine: `train_reinforce_baseline_hybrid_engine.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_hybrid_engine.sh>`_

Common Options (Shared Across Trainers)
---------------------------------------

Common CLI options shared across OpenRLHF trainers are documented in :doc:`common_options`.


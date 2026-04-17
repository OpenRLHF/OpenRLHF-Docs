Agent-based Training Guide
==========================

This page is the full recipe catalog for RL training in OpenRLHF — reward models, RL algorithms, single- and multi-turn agent modes, dynamic sampling, off-policy corrections, and VLM training. For the conceptual background see :doc:`agent_paradigm`; for shared CLI flags see :doc:`common_options`.

.. contents::
   :local:
   :depth: 2

.. _train_rm:

Reward Model (RM)
-----------------

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 --logging_steps 1 --eval_steps -1 \
      --train_batch_size 256 --micro_train_batch_size 4 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --param_dtype bf16 --max_epochs 1 --max_len 8192 \
      --zero_stage 3 --learning_rate 9e-6 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template --chosen_key chosen --rejected_key rejected \
      --attn_implementation flash_attention_2 \
      --packing_samples --gradient_checkpointing \
      --use_wandb {wandb_token}

RM-specific flags:

- ``--chosen_key`` / ``--rejected_key``: dataset JSON keys for the preference pair.
- ``--tokenizer_chat_template``: custom chat template.
- ``--value_head_prefix``: score-head prefix (defaults to ``score``).
- ``--packing_samples``: enable RM sample packing.

Setting ``--value_head_prefix score`` lets you later load the RM with ``AutoModelForSequenceClassification``:

.. code-block:: python

   reward_model = AutoModelForSequenceClassification.from_pretrained(
       reward_model_path,
       num_labels=1,
       torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
       use_cache=False,
   )
   # inputs: left-padded token IDs
   reward = reward_model.model(*inputs).last_hidden_state
   reward = reward_model.score(reward)[:, -1]

.. _rayppo:

RL Training (Ray + vLLM)
------------------------

All RL training runs through the unified agent execution pipeline (single-turn by default).

.. code-block:: bash

   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # additional worker nodes (optional)
   ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 --reward_num_gpus_per_node 8 \
      --critic_num_nodes 1 --critic_num_gpus_per_node 8 \
      --actor_num_nodes 1 --actor_num_gpus_per_node 8 \
      --vllm_num_engines 4 --vllm_tensor_parallel_size 2 \
      --colocate_all_models --vllm_gpu_memory_utilization 0.6 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
      --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
      --micro_train_batch_size 16 --train_batch_size 128 \
      --micro_rollout_batch_size 32 --rollout_batch_size 1024 \
      --max_samples 100000 --max_epochs 1 --max_len 2048 \
      --zero_stage 3 --param_dtype bf16 \
      --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages --apply_chat_template \
      --packing_samples --use_dynamic_batch \
      --normalize_reward --adam_offload \
      --attn_implementation flash_attention_2 \
      --gradient_checkpointing \
      --vllm_sync_backend nccl \
      --vllm_enable_sleep --deepspeed_enable_sleep \
      --use_wandb {wandb_token}

.. note::
   - Hybrid Engine (``--colocate_all_models``) is used here; see :doc:`hybrid_engine` for details.
   - Ray + vLLM does not currently support LoRA.
   - ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`` lets Ray auto-deploy the environment.
   - For GPU-index errors: ``export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` (NVIDIA) or ``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`` (AMD).

RL algorithms
-------------

Choose with ``--advantage_estimator``. All algorithms work with every execution mode.

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Algorithm
     - ``--advantage_estimator``
     - Key feature
     - Best for
   * - **PPO**
     - ``gae`` (default)
     - Full critic network
     - Stable training
   * - **REINFORCE++**
     - ``reinforce``
     - PPO tricks without critic
     - Less memory
   * - **REINFORCE++-baseline**
     - ``reinforce_baseline``
     - Mean-reward baseline
     - RLVR / reasoning
   * - **RLOO**
     - ``rloo``
     - Per-token KL + PPO clip
     - Multi-sample training
   * - **GRPO**
     - ``group_norm``
     - Group normalization
     - Batch-based training
   * - **Dr. GRPO**
     - ``dr_grpo``
     - Simplified GRPO
     - Removes local ``/std`` norm

Dynamic sampling
----------------

For reasoning tasks it is common to generate **multiple rollouts per prompt** and train on only a subset. Enable with:

- ``--n_samples_per_prompt``: completions per prompt (> 1 required).
- ``--dynamic_filtering``: DAPO-style filtering.
- ``--dynamic_filtering_reward_range <low> <high>``: reward range (e.g., ``0.0 1.0``); samples outside are dropped.
- ``--vllm_generate_batch_size``: oversampling (can exceed ``--rollout_batch_size``).
- ``--use_dynamic_batch``: pair with ``--train_max_tokens_per_gpu`` / ``--rollout_max_tokens_per_gpu`` to reduce padding waste.

Requires ``--n_samples_per_prompt > 1`` and either ``--remote_rm_url`` or ``--agent_func_path`` to produce ``scores`` in ``[0, 1]``. A common sizing choice is ``train_batch_size = rollout_batch_size * n_samples_per_prompt``; with filtering, the effective batch may be smaller.

Off-policy correction: TIS / ICEPOP / Seq-Mask-TIS
--------------------------------------------------

When rollouts are generated by vLLM, OpenRLHF can correct for the mismatch between rollout log-probs and training-time log-probs. Enable with ``--enable_vllm_is_correction`` and pick a strategy via ``--vllm_is_correction_type``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Strategy
     - Flag
     - Behavior
   * - **TIS** (default)
     - ``--vllm_is_correction_type tis``
     - Token-level clamp of the IS ratio into ``[low, high]``.
   * - **ICEPOP**
     - ``--vllm_is_correction_type icepop``
     - Token-level filter: zero out coefficients outside ``[low, high]``.
   * - **Seq-Mask-TIS**
     - ``--vllm_is_correction_type seq-mask-tis``
     - Sequence-level geometric-mean masking.

Thresholds: ``--vllm_is_truncated_threshold <low> <high>`` (default ``0.5 5.0``). Background: `off-policy RL training <https://fengyao.notion.site/off-policy-rl>`_.

ICEPOP is equivalent to masking:

.. code-block:: python

   # ICEPOP: zero-out coefficients outside the interval
   vllm_is = exp(old_log_probs - rollout_log_probs)
   mask = (vllm_is >= low) & (vllm_is <= high)
   vllm_is = vllm_is * mask

.. _single_turn_mode:

Single-turn mode (default)
--------------------------

Single-turn covers most RLHF use cases: one prompt → one response → one reward. You can use a remote RM server, a local Python reward function, or both.

Remote reward-model server
~~~~~~~~~~~~~~~~~~~~~~~~~~

Host a large RM behind an HTTP endpoint using ``openrlhf.cli.serve_rm``:

.. code-block:: bash

   python -m openrlhf.cli.serve_rm \
       --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
       --port 5000 \
       --param_dtype bf16 \
       --attn_implementation flash_attention_2 \
       --normalize_reward \
       --max_len 8192 --batch_size 16

Then pass its URL to the trainer:

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
       --remote_rm_url http://localhost:5000/get_reward \
       ... # other training args

Custom reward function (RFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a local Python file to ``--remote_rm_url``; OpenRLHF will call it on-the-fly. Use ``--label_key`` to forward a dataset field as the ground-truth label. See `train_ppo_with_reward_fn.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh>`_ for an end-to-end example.

.. _multi_turn_mode:

Multi-turn mode
---------------

For multi-step interactions (reasoning chains, coding with feedback, game playing, tool use), use multi-turn agent execution via ``--agent_func_path``. Each sample is an *episode*:

1. Reset environment with the initial prompt/label.
2. Model generates an action (text).
3. Environment returns feedback, an optional reward, and ``done``.
4. Repeat until ``done=True``.

Agent implementation
~~~~~~~~~~~~~~~~~~~~

Implement ``AgentInstanceBase`` (and optionally wrap it with ``MultiTurnAgentExecutor``):

.. code-block:: python

   # agent_func.py
   from typing import Any, Dict
   import torch
   from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


   class AgentInstance(AgentInstanceBase):
       def __init__(self, *args, **kwargs):
           self.step_idx = 0

       async def reset(self, states: dict, **kwargs) -> dict:
           # states typically contains: observation (prompt), label (ground truth)
           self.step_idx = 0
           return {"observation": states["observation"]}

       async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
           # states typically contains:
           #   observation_text: current context
           #   action_text: model output for this step
           #   label: ground-truth label (from --label_key)
           self.step_idx += 1
           done = self.step_idx >= 3

           # Reward usually only at the terminal step
           reward = torch.tensor(1.0) if done else torch.tensor(0.0)

           environment_feedback = (
               "\n\nHuman: [CORRECT]\n</s>"
               if done
               else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
           )

           return {
               "rewards": reward,               # advantage calculation
               "scores": reward,                # dynamic filtering
               "environment_feedback": environment_feedback,
               "done": done,
               "sampling_params": states.get("sampling_params", None),
               "extra_logs": {"step": self.step_idx},
           }


   class AgentExecutor(MultiTurnAgentExecutor):
       def __init__(self):
           super().__init__(AgentInstance)

Return schema:

- ``reset()`` returns ``{"observation": str}``.
- ``step()`` returns a dict with (at least) ``rewards`` (Tensor), ``scores`` (Tensor), ``environment_feedback`` (str), ``done`` (bool). Optional: ``sampling_params`` (per-step vLLM params), ``extra_logs`` (dict of metrics).

Launching
~~~~~~~~~

Synchronous (more stable) vs asynchronous (higher throughput, may affect convergence):

.. code-block:: bash

   # synchronous
   python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /path/to/agent_func.py \
      ... # other training args

   # asynchronous
   export VLLM_USE_V1=1
   python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --agent_func_path /path/to/agent_func.py \
      --async_train \
      ... # other training args

Multi-turn / async flags:

- ``--agent_func_path``: path to your multi-turn agent file.
- ``--async_train``: enable async training (mutually exclusive with ``--colocate_all_models``).
- ``--async_queue_size``: async buffer size (larger = more off-policy; default ``1``).
- ``--partial_rollout``: requires ``--async_train``. Uses vLLM pause/resume for weight sync, so generation overlaps with training. In-flight samples may contain tokens from both old and new weights.

.. warning::
   Async training may affect training stability. Use it when throughput is critical and convergence has been validated.

OpenAI-compatible Agent Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your agent needs an OpenAI-style chat API (e.g., to integrate external tool-use frameworks), `examples/python/agent_func_openai_server_executor.py <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_openai_server_executor.py>`_ wraps vLLM as a local ``/v1/chat/completions`` server while still collecting token-level traces for RL training.

- Exposes ``/v1/chat/completions``, ``/v1/models``, ``/tokenize``.
- Collects token IDs and logprobs per session.
- Delta-tokenization reuses prefix tokens across multi-turn calls.
- Override ``run_agent()`` to plug in your own workflow.

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --agent_func_path examples/python/agent_func_openai_server_executor.py \
      ... # other training args

Upstream references:

- Async agent RLHF: `train_reinforce_baseline_ray_agent_async.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_
- REINFORCE++-baseline + Hybrid Engine: `train_reinforce_baseline_hybrid_engine.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_hybrid_engine.sh>`_

Vision-Language Model (VLM) RLHF
--------------------------------

Since OpenRLHF 0.10, VLMs (e.g., Qwen3.5) can be trained end-to-end with image inputs. VLMs are auto-detected via the ``vision_config`` in the HuggingFace config, loaded with ``AutoModelForImageTextToText``; ``AutoProcessor`` handles image-token insertion, and images are forwarded to vLLM for multimodal generation.

VLM flags:

- ``--image_key``: dataset key for image paths/URLs (default ``images``).
- ``--max_images_per_prompt``: max images per prompt for vLLM (``0`` = text-only; default ``0``).
- ``--freeze_visual_encoder``: freeze the vision encoder; only sync language-model weights to vLLM.

Dataset format (JSONL):

.. code-block:: json

   {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Find x."}]}],
    "images": ["/path/to/image.png"],
    "label": "3"}

Example — see `train_vlm_math_hybrid_engine.sh <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_vlm_math_hybrid_engine.sh>`_:

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3.5-2B \
      --remote_rm_url examples/python/math_reward_func.py \
      --prompt_data hiyouga/geometry3k \
      --input_key prompt --label_key label \
      --image_key images --max_images_per_prompt 1 \
      --freeze_visual_encoder \
      --max_len 4096 --max_new_tokens 2048 \
      --advantage_estimator reinforce_baseline \
      --colocate_all_models --vllm_gpu_memory_utilization 0.7 \
      --apply_chat_template \
      --attn_implementation eager \
      --param_dtype bf16 \
      ... # other training args

.. note::
   Tested with Qwen3.5 (hybrid linear + full attention). Other VLMs with a ``ForConditionalGeneration`` architecture (Gemma4, LLaVA, InternVL, ...) are auto-detected but not yet tested. Use ``--attn_implementation eager`` for models with linear attention layers — flash attention may not support packed sequences there.

Multi-turn VLM RLHF is also supported; see `vlm_multiturn_agent.py <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/vlm_multiturn_agent.py>`_.

Shared CLI flags across trainers are documented in :doc:`common_options`.

RL Training Guide
=================

This page is the complete reference for **RL training** in OpenRLHF. Every algorithm and execution mode runs through the same agent execution pipeline. For supervised methods (SFT, RM, DPO) see :doc:`non_rl`. For the conceptual model see :doc:`agent_paradigm`. For shared CLI flags see :doc:`common_options`. For sync vs async pipelines see :doc:`hybrid_engine` and :doc:`async_training`.

.. contents::
   :local:
   :depth: 2

Overview
--------

Every RL run in OpenRLHF combines three orthogonal choices:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Axis
     - Options
     - How to set
   * - **RL algorithm**
     - PPO / REINFORCE++ / REINFORCE++-baseline / RLOO / GRPO / Dr. GRPO
     - ``--advantage_estimator``
   * - **Execution mode**
     - Single-turn (default) | Multi-turn
     - default / ``--remote_rm_url`` | ``--agent_func_path``
   * - **Pipeline**
     - Sync (Hybrid Engine) | Async (with optional partial rollout)
     - :doc:`hybrid_engine` | :doc:`async_training`

These axes are independent: any algorithm runs in any mode under any pipeline, because every rollout produces token-level trajectories that are consumed identically by the loss layer.

.. _rayppo:

Quick Launch (Ray + vLLM)
-------------------------

The default RL example throughout the docs is **Qwen3-4B-Thinking trained with REINFORCE++-baseline
on math (RLVR)**. The full launch command is documented in :doc:`hybrid_engine`. Here is the
minimal version with the essential flags only:

.. code-block:: bash

   # launch the master node of ray in a container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3-4B-Thinking-2507 \
      --remote_rm_url examples/python/math_reward_func.py \
      --prompt_data zhuzilin/dapo-math-17k \
      --input_key prompt \
      --label_key label \
      --apply_chat_template \
      --packing_samples \
      \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 4 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.7 \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep \
      --vllm_sync_backend nccl \
      --enforce_eager \
      \
      --advantage_estimator reinforce_baseline \
      --use_kl_loss \
      --kl_estimator k2 \
      --init_kl_coef 1e-5 \
      --enable_vllm_is_correction \
      --vllm_is_correction_type icepop \
      \
      --rollout_batch_size 128 \
      --n_samples_per_prompt 8 \
      --train_batch_size 1024 \
      --dynamic_filtering \
      --dynamic_filtering_reward_range 0.0 1.0 \
      --use_dynamic_batch \
      --max_len 74240 \
      --max_new_tokens 64000 \
      \
      --zero_stage 3 \
      --param_dtype bf16 \
      --gradient_checkpointing \
      --actor_learning_rate 5e-7 \
      --save_path ./exp/Qwen3-4B-Thinking

.. note::

   - Hybrid Engine is used here; for the configuration deep-dive see :doc:`hybrid_engine`.
   - Ray + vLLM does **not** currently support LoRA.
   - Auto-deploy environment to Ray workers:
     ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'``.
   - GPU-index issues: ``export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` (NVIDIA)
     or ``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`` (AMD).

RL Algorithms
-------------

Choose with ``--advantage_estimator``. Algorithms differ in **whether they use a critic**, **how they baseline the reward**, and **how they normalize advantages**. All work with every execution mode and pipeline.

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - Algorithm
     - ``--advantage_estimator``
     - Key idea
     - Best for
   * - **PPO**
     - ``gae`` *(default)*
     - GAE with full critic; clipped surrogate objective
     - General RLHF, stable training
   * - **REINFORCE++**
     - ``reinforce``
     - Critic-free; PPO clip, KL penalty, reward normalization
     - Lower memory than PPO
   * - **REINFORCE++-baseline**
     - ``reinforce_baseline``
     - Mean reward as baseline (no critic, no per-prompt std)
     - **RLVR / reasoning** — robust to reward scale
   * - **RLOO**
     - ``rloo``
     - Leave-one-out baseline + PPO clip + per-token KL
     - Multi-sample-per-prompt training
   * - **GRPO**
     - ``group_norm``
     - Per-group mean/std normalization + KL loss term
     - Batch-based reasoning training
   * - **Dr. GRPO**
     - ``dr_grpo``
     - GRPO without local ``/std`` normalization
     - When per-group std normalization hurts

Algorithm-specific requirements:

- ``rloo`` / ``reinforce_baseline`` / ``group_norm`` require ``--n_samples_per_prompt > 1``.
- ``group_norm`` (GRPO) typically pairs with ``--use_kl_loss`` and ``--kl_estimator k3``.

PPO loss & clipping
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--eps_clip``
     - PPO clip range (default ``0.2``).
   * - ``--eps_clip_low_high <low> <high>``
     - Asymmetric clip bounds; overrides ``--eps_clip`` when set.
   * - ``--dual_clip <c>``
     - Dual-clip PPO upper bound (typical ``c=3``); prevents oversized policy updates on negative advantages.
   * - ``--value_clip``
     - Critic value clip (default ``0.5``).
   * - ``--policy_loss_type {ppo, gspo}``
     - Switch between standard PPO and GSPO-style loss aggregation.
   * - ``--ptx_coef``
     - PPO-ptx pre-training loss coefficient (default ``0.05``).

Advantage / GAE
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--gamma``
     - Discount factor (default ``1.0`` — treats trajectory as one episode, no discounting).
   * - ``--lambd``
     - GAE λ (default ``1.0``); lower λ trades bias for variance.
   * - ``--no_advantage_std_norm``
     - Keep mean centering but disable dividing by advantage std.
   * - ``--normalize_reward``
     - Online running mean/std normalization of raw rewards.
   * - ``--reward_clip_range <low> <high>``
     - Clamp raw rewards before advantage computation (default ``-10 10``).

KL control
~~~~~~~~~~

KL divergence between the current policy and the reference policy can be applied either as a **penalty on the reward** or as a **separate loss term**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``--init_kl_coef``
     - Initial KL penalty coefficient (default ``0.01``). Set ``0`` to disable the reference model entirely.
   * - ``--kl_target`` / ``--kl_horizon``
     - Adaptive KL controller — when ``--kl_target`` is set, the coefficient adapts toward this target over ``--kl_horizon`` steps.
   * - ``--use_kl_loss``
     - Add KL as a separate loss term (GRPO-style) rather than a reward penalty.
   * - ``--kl_estimator {k1, k2, k3}``
     - KL estimator: ``k1`` for standard PPO penalty, ``k2`` ≈ ``k1`` when used as loss, ``k3`` for GRPO loss.

Recommended pairings:

- Standard PPO / REINFORCE++: ``--init_kl_coef 0.01`` (penalty), ``--kl_estimator k1``.
- GRPO: ``--use_kl_loss --kl_estimator k3``.

Entropy
~~~~~~~

- ``--entropy_loss_coef``: entropy regularization coefficient. ``None`` disables it; ``0`` only logs entropy without applying it as a loss.

Length penalties (DAPO / ProRL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reasoning workloads where models can run away with output length:

- ``--overlong_buffer_len <L>``: penalize responses whose length exceeds ``max_new_tokens - L`` (DAPO-style soft length limit).
- ``--overlong_penalty_factor``: multiplicative penalty magnitude (default ``1.0``).
- ``--stop_properly_penalty_coef <c>``: penalty for samples truncated by length (``finish_reason='length'``).
  ``c >= 0`` scales the reward by ``c ∈ [0, 1]``; ``c < 0`` overrides the reward (e.g., ``-0.5``).

Misc
~~~~

- ``--freezing_actor_steps``: keep actor frozen for the first *N* updates while critic warms up.
- ``--top_p`` / ``--temperature``: vLLM sampling parameters during rollouts.
- ``--save_value_network``: also save the critic checkpoint (PPO only).
- ``--full_determinism``: enable bit-reproducible behavior (slower; vLLM v1 + fixed seed paths).

Dynamic Sampling (DAPO)
-----------------------

For reasoning tasks, generate **multiple completions per prompt** and train only on a subset:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--n_samples_per_prompt``
     - Completions per prompt (must be > 1 for filtering / RLOO / GRPO / REINFORCE++-baseline).
   * - ``--dynamic_filtering``
     - Enable DAPO-style filtering by ``scores`` returned from the reward / agent function.
   * - ``--dynamic_filtering_reward_range <low> <high>``
     - Reward range to keep, e.g., ``0.0 1.0``. Samples outside the range are dropped.
   * - ``--vllm_generate_batch_size``
     - vLLM generation batch size; can exceed ``--rollout_batch_size`` for oversampling.
       Requires ``--async_train`` when greater than ``--rollout_batch_size``.
   * - ``--use_dynamic_batch``
     - Form micro-batches by token budget instead of count — much better packing for variable-length sequences.
       Pair with ``--train_max_tokens_per_gpu`` and ``--rollout_max_tokens_per_gpu``.

Sizing rule of thumb: ``train_batch_size = rollout_batch_size * n_samples_per_prompt``. With
``--dynamic_filtering`` the effective batch may shrink if many samples are filtered out — keep
oversample headroom via ``--vllm_generate_batch_size`` (async only).

End-to-end DAPO recipe: `train_dapo_ray_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_dapo_ray_hybrid_engine.sh>`_.

Off-policy correction (TIS / ICEPOP / Seq-Mask-TIS)
---------------------------------------------------

Because vLLM uses different kernels (and sometimes a different precision) than the trainer, the
same token sequence can produce slightly different log-probs in rollout vs. training. OpenRLHF can
apply **importance-sampling correction** to compensate.

Enable with ``--enable_vllm_is_correction`` and pick a strategy:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Strategy
     - Flag
     - Behavior
   * - **TIS** *(default)*
     - ``--vllm_is_correction_type tis``
     - Token-level **clamp** of the IS ratio into ``[low, high]``.
   * - **ICEPOP**
     - ``--vllm_is_correction_type icepop``
     - Token-level **filter** — zero out coefficients outside ``[low, high]`` (no clamp).
   * - **Seq-Mask-TIS**
     - ``--vllm_is_correction_type seq-mask-tis``
     - Sequence-level geometric-mean masking.

Thresholds via ``--vllm_is_truncated_threshold <low> <high>`` (default ``0.5 5.0``).
Background: `off-policy RL training <https://fengyao.notion.site/off-policy-rl>`_.

ICEPOP is equivalent to a hard mask:

.. code-block:: python

   # ICEPOP: zero-out coefficients outside the interval
   vllm_is = exp(old_log_probs - rollout_log_probs)
   mask = (vllm_is >= low) & (vllm_is <= high)
   vllm_is = vllm_is * mask

.. tip::
   Async + partial rollout pairs naturally with ICEPOP because in-flight samples mix old and new
   weights. See :doc:`async_training`.

.. _single_turn_mode:

Single-Turn Mode (Default)
--------------------------

Single-turn covers the vast majority of RLHF use cases: one prompt → one response → one reward.
The reward source can be any of:

1. A trained reward model (``--reward_pretrain``).
2. A remote HTTP RM server (``--remote_rm_url http://host:5000/get_reward``).
3. A local Python reward function (``--remote_rm_url /path/to/reward_func.py``) —
   this enables **Reinforced Fine-Tuning (RFT)** for code, math, formatting, etc.

Remote reward-model server
~~~~~~~~~~~~~~~~~~~~~~~~~~

Host a large RM behind an HTTP endpoint:

.. code-block:: bash

   python -m openrlhf.cli.serve_rm \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --port 5000 \
      --param_dtype bf16 \
      --attn_implementation flash_attention_2 \
      --normalize_reward \
      --max_len 8192 \
      --batch_size 16

Then point the trainer at it:

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --remote_rm_url http://localhost:5000/get_reward \
      ...

Custom reward function (Reinforced Fine-Tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a Python file path to ``--remote_rm_url``; OpenRLHF imports and calls it on the fly:

.. code-block:: python

   # reward_func.py
   import torch

   def reward_func(queries, prompts, labels):
       """
       Args:
           queries: list[str] — full text (prompt + response) per sample
           prompts: list[str] — original prompts
           labels:  list[str] — ground-truth labels (from --label_key)

       Returns:
           dict with:
               rewards:    Tensor — used in advantage calculation
               scores:     Tensor — used by --dynamic_filtering (typically in [0, 1])
               extra_logs: dict   — values logged to Wandb / TensorBoard
       """
       batch_size = len(queries)
       reward = torch.zeros(batch_size)
       for i, (q, label) in enumerate(zip(queries, labels)):
           reward[i] = 1.0 if my_check(q, label) else 0.0
       return {
           "rewards": reward,
           "scores": reward,
           "extra_logs": {"accuracy": reward.mean().item()},
       }

Pass the ground-truth field via ``--label_key answer`` (or whichever JSON key holds your label).
End-to-end example: `train_ppo_with_reward_fn.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh>`_.

.. tip::
   Typical RFT use cases: code (run unit tests), math (verify final answer), JSON formatting
   (regex check), multi-objective rewards (combine signals).

.. _multi_turn_mode:

Multi-Turn Mode
---------------

For multi-step interactions — reasoning chains, coding with feedback, game playing, tool use —
implement a multi-turn agent and pass it via ``--agent_func_path``. Each sample becomes an
*episode*:

1. **Reset** the environment with the initial prompt + label.
2. The model generates an action.
3. The environment returns feedback, an optional reward, and ``done``.
4. Repeat until ``done=True``.

OpenRLHF wraps everything in the same token-in-token-out trajectory consumed by the loss, so any
RL algorithm just works.

Implementing an agent
~~~~~~~~~~~~~~~~~~~~~

Subclass ``AgentInstanceBase`` and (optionally) wrap it in ``MultiTurnAgentExecutor``:

.. code-block:: python

   # agent_func.py
   from typing import Any, Dict
   import torch
   from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


   class AgentInstance(AgentInstanceBase):
       def __init__(self, *args, **kwargs):
           self.step_idx = 0

       async def reset(self, states: dict, **kwargs) -> dict:
           # states: {"observation": <prompt>, "label": <ground truth>, ...}
           self.step_idx = 0
           return {"observation": states["observation"]}

       async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
           # states: observation_text / action_text / label / sampling_params / ...
           self.step_idx += 1
           done = self.step_idx >= 3
           reward = torch.tensor(1.0) if done else torch.tensor(0.0)

           feedback = (
               "\n\nHuman: [CORRECT]\n</s>"
               if done
               else "\n\nHuman: [INCORRECT]\nPlease analyze and try again.\n</s>\n\nAssistant: "
           )

           return {
               "rewards": reward,                 # used for advantage
               "scores": reward,                  # used for dynamic filtering
               "environment_feedback": feedback,  # appended to next-turn context
               "done": done,
               "sampling_params": states.get("sampling_params"),
               "extra_logs": {"step": self.step_idx},
           }


   class AgentExecutor(MultiTurnAgentExecutor):
       def __init__(self):
           super().__init__(AgentInstance)

Return-value contract:

- ``reset(states)`` → ``{"observation": str}``.
- ``step(states)`` → dict containing ``rewards`` (Tensor), ``scores`` (Tensor),
  ``environment_feedback`` (str), ``done`` (bool). Optional: ``sampling_params``
  (per-step vLLM params), ``environment_images`` (for VLM agents),
  ``extra_logs`` (metric dict).

For complete custom token-level control, subclass ``AgentExecutorBase`` directly and implement
``execute()`` — but stick to the **token-in-token-out principle** so sampling and training stay
aligned.

Launching multi-turn training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3-4B-Thinking-2507 \
      --agent_func_path /path/to/agent_func.py \
      ...

For higher throughput add ``--async_train`` (and optionally ``--partial_rollout``); see
:doc:`async_training` for the configuration.

OpenAI-compatible Agent Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your agent needs an OpenAI-style chat API (e.g., to plug into existing tool-use frameworks),
`examples/python/agent_func_openai_server_executor.py
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_openai_server_executor.py>`_
wraps the local vLLM as a ``/v1/chat/completions`` server while still collecting token-level
traces for RL training:

- Standard endpoints: ``/v1/chat/completions``, ``/v1/models``, ``/tokenize``.
- Per-session token IDs and logprobs are captured automatically.
- Delta-tokenization reuses prefix tokens across multi-turn calls.
- Override ``run_agent()`` to plug in your own multi-turn workflow.

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --agent_func_path examples/python/agent_func_openai_server_executor.py \
      ...

Upstream references:

- Async agent RLHF: `train_reinforce_baseline_ray_agent_async.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_
- REINFORCE++-baseline + Hybrid Engine: `train_reinforce_baseline_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_hybrid_engine.sh>`_
- DAPO with dynamic filtering: `train_dapo_ray_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_dapo_ray_hybrid_engine.sh>`_
- ProRL V2 reasoning recipe: `train_prorlv2_math_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_prorlv2_math_hybrid_engine.sh>`_
- Custom Python reward (RFT): `train_ppo_with_reward_fn.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh>`_

Vision-Language Model (VLM) RLHF
--------------------------------

Since OpenRLHF 0.10, VLMs (e.g., **Qwen3.5**) can be trained end-to-end with image inputs. The
framework auto-detects VLMs via the ``vision_config`` field in the HuggingFace config, loads them
with ``AutoModelForImageTextToText``, uses ``AutoProcessor`` for correct image-token insertion,
and forwards images to vLLM for multimodal generation.

Why this matters: previous VLM RLHF setups required custom data loaders and bespoke inference
paths. OpenRLHF reuses the **same agent execution pipeline**, the **same RL algorithms**, and the
**same Hybrid Engine** for VLMs as for text-only models — you only add a few flags.

VLM-specific flags:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--image_key``
     - Dataset JSON key holding the image paths/URLs (default ``images``).
   * - ``--max_images_per_prompt``
     - Max images per prompt for vLLM (``0`` = text-only; default ``0``).
   * - ``--freeze_visual_encoder``
     - Freeze the vision encoder; only language-model weights are trained and synced to vLLM
       (saves memory and weight-sync time).

Dataset format (JSONL):

.. code-block:: json

   {
       "prompt": [
           {"role": "user", "content": [
               {"type": "image"},
               {"type": "text", "text": "Find x."}
           ]}
       ],
       "images": ["/path/to/image.png"],
       "label": "3"
   }

End-to-end example — see `train_vlm_math_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_vlm_math_hybrid_engine.sh>`_:

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3.5-2B \
      --remote_rm_url examples/python/math_reward_func.py \
      --prompt_data hiyouga/geometry3k \
      --input_key prompt \
      --label_key label \
      --image_key images \
      --max_images_per_prompt 1 \
      --freeze_visual_encoder \
      --max_len 4096 \
      --max_new_tokens 2048 \
      --advantage_estimator reinforce_baseline \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.7 \
      --apply_chat_template \
      --attn_implementation eager \
      --param_dtype bf16 \
      ...

.. note::
   - **Tested**: Qwen3.5 (hybrid linear + full attention).
   - **Auto-detected, not yet tested**: VLMs with a ``ForConditionalGeneration`` architecture
     (Gemma4, LLaVA, InternVL, ...).
   - Use ``--attn_implementation eager`` for models with linear attention layers — flash attention
     may not support packed sequences there.
   - Multi-turn VLM RLHF is supported; see
     `vlm_multiturn_agent.py
     <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/vlm_multiturn_agent.py>`_.

Logging & evaluation
--------------------

- ``--use_wandb {token}`` / ``--wandb_project`` / ``--wandb_group`` / ``--wandb_run_name``:
  Wandb logging.
- ``--use_tensorboard {logdir}``: TensorBoard logging.
- ``--logging_steps``: log every *N* training steps.
- ``--eval_steps`` / ``--eval_dataset``: periodic evaluation on a held-out dataset.
- ``--num_episodes``: total RL episodes to run (one episode = one full rollout pass through
  ``rollout_batch_size`` prompts).

Training metrics include policy loss, KL, entropy, reward / advantage statistics, generation
length, grad norm, and per-phase wall-clock time. See :doc:`checkpoint` for save/resume mechanics
and :doc:`performance` for tuning.

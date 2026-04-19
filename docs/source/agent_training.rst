RL Training Guide
=================

This page is the complete reference for **RL training** in OpenRLHF. Every algorithm and execution
mode runs through the same agent execution pipeline.

For supervised methods (SFT, RM, DPO) see :doc:`non_rl`. For the conceptual model see
:doc:`agent_paradigm`. For shared CLI flags see :doc:`common_options`. For sync vs. async pipelines
see :doc:`hybrid_engine` and :doc:`async_training`.

.. note::
   All flags shown on this page use the **0.10.2 hierarchical CLI**. Entity config lives under
   ``--actor.*`` / ``--critic.*`` / ``--ref.*`` / ``--reward.*``; pipeline config under
   ``--ds.*`` / ``--vllm.*`` / ``--rollout.*`` / ``--data.*`` / ``--train.*`` / ``--eval.*`` /
   ``--ckpt.*`` / ``--logger.*`` / ``--algo.*``. Old flat flags like ``--pretrain`` or
   ``--remote_rm_url`` no longer parse — see :ref:`flag_migration`.

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
   * - **Execution mode**
     - Single-turn (default) | Multi-turn
     - default / ``--reward.remote_url`` | ``--train.agent_func_path``
   * - **RL algorithm**
     - PPO / REINFORCE++ / REINFORCE++-baseline / RLOO / GRPO / Dr. GRPO
     - ``--algo.advantage.estimator``
   * - **Pipeline**
     - Sync (Hybrid Engine) | Async (with optional partial rollout)
     - :doc:`hybrid_engine` | :doc:`async_training`

These axes are independent: any algorithm runs in any mode under any pipeline, because every
rollout produces token-level trajectories that are consumed identically by the loss layer.

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
      --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
      --reward.remote_url examples/python/math_reward_func.py \
      --data.prompt_dataset zhuzilin/dapo-math-17k \
      --data.input_key prompt \
      --data.label_key label \
      --data.apply_chat_template \
      --ds.packing_samples \
      \
      --ref.num_nodes 1 \
      --ref.num_gpus_per_node 4 \
      --actor.num_nodes 1 \
      --actor.num_gpus_per_node 4 \
      --vllm.num_engines 2 \
      --vllm.tensor_parallel_size 2 \
      --train.colocate_all \
      --vllm.gpu_memory_utilization 0.7 \
      --vllm.enable_sleep \
      --ds.enable_sleep \
      --vllm.sync_backend nccl \
      --vllm.enforce_eager \
      \
      --algo.advantage.estimator reinforce_baseline \
      --algo.kl.use_loss \
      --algo.kl.estimator k2 \
      --algo.kl.init_coef 1e-5 \
      --actor.entropy_coef 0.0 \
      --algo.advantage.is_correction_enable \
      --algo.advantage.is_correction_type icepop \
      \
      --rollout.batch_size 128 \
      --rollout.n_samples_per_prompt 8 \
      --train.batch_size 1024 \
      --algo.dynamic_filtering_enable \
      --algo.dynamic_filtering_range 0.0 1.0 \
      --train.dynamic_batch_enable \
      --data.max_len 74240 \
      --rollout.max_new_tokens 64000 \
      \
      --ds.zero_stage 3 \
      --ds.param_dtype bf16 \
      --actor.gradient_checkpointing_enable \
      --actor.adam.lr 5e-7 \
      --ckpt.output_dir ./exp/Qwen3-4B-Thinking

.. note::

   - Hybrid Engine is used here; for the configuration deep-dive see :doc:`hybrid_engine`.
   - Ray + vLLM does **not** currently support LoRA.
   - Auto-deploy environment to Ray workers:
     ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'``.
   - GPU-index issues: ``export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` (NVIDIA)
     or ``RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`` (AMD).

Execution Modes
---------------

The execution mode determines **how an experience is collected**. Choose based on the structure of
your task.

.. _single_turn_mode:

Single-Turn Mode (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

One prompt → one response → one reward. Covers the vast majority of RLHF use cases. The reward
source is one of:

1. A **trained reward model** — set ``--reward.model_name_or_path``.
2. A **remote HTTP RM server** — set ``--reward.remote_url http://host:5000/get_reward``.
3. A **local Python reward function** — set ``--reward.remote_url /path/to/reward_func.py``.
   This enables Reinforced Fine-Tuning (RFT) for code, math, formatting, multi-objective
   rewards, etc.

**Remote reward-model server.** Host a large RM behind an HTTP endpoint:

.. code-block:: bash

   python -m openrlhf.cli.serve_rm \
      --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
      --port 5000 \
      --ds.param_dtype bf16 \
      --ds.attn_implementation flash_attention_2 \
      --reward.normalize_enable \
      --data.max_len 8192 \
      --batch_size 16

   # then in the trainer:
   python3 -m openrlhf.cli.train_ppo_ray \
      --reward.remote_url http://localhost:5000/get_reward \
      ...

**Custom reward function (RFT).** Pass a Python file path to ``--reward.remote_url``; OpenRLHF
imports and calls it on the fly. Pass the ground-truth field via ``--data.label_key answer``:

.. code-block:: python

   # reward_func.py
   import torch

   def reward_func(queries, prompts, labels):
       """
       Args:
           queries: list[str] — full text (prompt + response) per sample
           prompts: list[str] — original prompts
           labels:  list[str] — ground-truth labels (from --data.label_key)

       Returns:
           dict with:
               rewards:    Tensor — used in advantage calculation
               scores:     Tensor — used by --algo.dynamic_filtering_enable (typically in [0, 1])
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

End-to-end example: `train_ppo_with_reward_fn.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh>`_.

.. tip::
   Typical RFT use cases: code (run unit tests), math (verify final answer), JSON formatting
   (regex check), multi-objective rewards (combine signals).

.. _multi_turn_mode:

Multi-Turn Mode
~~~~~~~~~~~~~~~

For multi-step interactions — reasoning chains, coding with feedback, game playing, tool use —
implement a multi-turn agent and pass it via ``--train.agent_func_path``. Each sample becomes an
*episode*:

1. **Reset** the environment with the initial prompt + label.
2. The model generates an action.
3. The environment returns feedback, an optional reward, and ``done``.
4. Repeat until ``done=True``.

OpenRLHF wraps everything in the same token-in-token-out trajectory consumed by the loss, so any
RL algorithm just works.

**Implementing an agent.** Subclass ``AgentInstanceBase`` and (optionally) wrap it in
``MultiTurnAgentExecutor``:

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

**Return-value contract:**

- ``reset(states)`` → ``{"observation": str}``.
- ``step(states)`` → dict containing ``rewards`` (Tensor), ``scores`` (Tensor),
  ``environment_feedback`` (str), ``done`` (bool). Optional: ``sampling_params``
  (per-step vLLM params), ``environment_images`` (for VLM agents),
  ``extra_logs`` (metric dict).

For complete custom token-level control, subclass ``AgentExecutorBase`` directly and implement
``execute()`` — but stick to the **token-in-token-out principle** so sampling and training stay
aligned.

**Launching:**

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --actor.model_name_or_path Qwen/Qwen3-4B-Thinking-2507 \
      --train.agent_func_path /path/to/agent_func.py \
      ...

For higher throughput add ``--train.async_enable`` (and optionally
``--train.partial_rollout_enable``); see :doc:`async_training`.

**OpenAI-compatible Agent Server.** When your agent needs an OpenAI-style chat API (e.g., to plug
into existing tool-use frameworks), `examples/python/agent_func_openai_server_executor.py
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_openai_server_executor.py>`_
wraps the local vLLM as a ``/v1/chat/completions`` server while still collecting token-level traces
for RL training:

- Standard endpoints: ``/v1/chat/completions``, ``/v1/models``, ``/tokenize``.
- Per-session token IDs and logprobs are captured automatically.
- Delta-tokenization reuses prefix tokens across multi-turn calls.
- Override ``run_agent()`` to plug in your own multi-turn workflow.

.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
      --train.agent_func_path examples/python/agent_func_openai_server_executor.py \
      ...

RL Algorithms
-------------

The algorithm determines **how the policy is updated** from the collected trajectories. Choose with
``--algo.advantage.estimator``. Algorithms differ in whether they use a critic, how they baseline
the reward, and how they normalize advantages. **All algorithms work with every execution mode and
pipeline.**

.. list-table::
   :header-rows: 1
   :widths: 22 22 28 28

   * - Algorithm
     - ``--algo.advantage.estimator``
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

- ``rloo`` / ``reinforce_baseline`` / ``group_norm`` require ``--rollout.n_samples_per_prompt > 1``.
- ``group_norm`` (GRPO) typically pairs with ``--algo.kl.use_loss`` and ``--algo.kl.estimator k3``.

Optimizer: Adam or Muon (per entity)
------------------------------------

PPO has two independently-optimized models (actor and critic). 0.10.2 lets you pick the
optimizer per entity:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--actor.optim {adam, muon}``
     - Actor optimizer (default ``adam``).
   * - ``--critic.optim {adam, muon}``
     - Critic optimizer (default ``adam``).
   * - ``--actor.muon.lr`` / ``--actor.muon.momentum``
     - Muon 2-D-weight group settings for the actor.
   * - ``--actor.adam.lr``
     - Adam LR when ``--actor.optim adam``; also the LR for Muon's aux-Adam subgroup when
       ``--actor.optim muon`` (embeddings / LM head / 1-D params).
   * - ``--actor.adam.betas`` / ``--actor.adam.eps`` / ``--actor.adam.weight_decay``
     - AdamW hyperparameters for the actor.
   * - ``--actor.lr_scheduler`` / ``--actor.lr_warmup_ratio`` / ``--actor.min_lr_ratio`` /
       ``--actor.max_norm``
     - Per-entity scheduler + gradient clip. Replace ``actor`` with ``critic`` for the critic
       side; they are fully independent.

Typical "actor-Muon, critic-Adam" combo (one-line in 0.10.2):

.. code-block:: bash

   --actor.optim muon \
   --actor.muon.lr 0.02 --actor.muon.momentum 0.95 \
   --actor.adam.lr 5e-7 \
   --critic.optim adam \
   --critic.adam.lr 9e-6

Muon requires DeepSpeed ≥ 0.18.2 and is **incompatible** with ``--ds.adam_offload``. See the
detailed caveats in :doc:`common_options` (``ns_steps`` / Nesterov placeholders, weight-decay
semantics).

Tuning
------

The flags below are tuning knobs around the chosen algorithm. Most users will only touch the KL
control and length-penalty knobs in practice.

Loss & clipping
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--actor.eps_clip``
     - PPO clip range (default ``0.2``).
   * - ``--actor.eps_clip_low_high <low> <high>``
     - Asymmetric clip bounds; overrides ``--actor.eps_clip`` when set.
   * - ``--actor.dual_clip <c>``
     - Dual-clip PPO upper bound (typical ``c=3``); prevents oversized policy updates on
       negative advantages.
   * - ``--critic.value_clip``
     - Critic value clip (default ``0.5``).
   * - ``--actor.policy_loss_type {ppo, gspo}``
     - Switch between standard PPO and GSPO-style loss aggregation.

Advantage / GAE
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--algo.advantage.gamma``
     - Discount factor (default ``1.0`` — treats trajectory as one episode, no discounting).
   * - ``--algo.advantage.lambd``
     - GAE λ (default ``1.0``); lower λ trades bias for variance.
   * - ``--algo.advantage.no_std_norm``
     - Keep mean centering but disable dividing by advantage std.

KL control
~~~~~~~~~~

KL divergence between the current policy and the reference policy can be applied either as a
**penalty on the reward** or as a **separate loss term**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Meaning
   * - ``--algo.kl.init_coef``
     - Initial KL coefficient (default ``0.01``). Set ``0`` to disable the reference model entirely.
   * - ``--algo.kl.target`` / ``--algo.kl.horizon``
     - Adaptive KL controller — when ``--algo.kl.target`` is set, the coefficient adapts toward
       this target over ``--algo.kl.horizon`` steps.
   * - ``--algo.kl.use_loss``
     - Add KL as a separate loss term (GRPO-style) rather than a reward penalty.
   * - ``--algo.kl.estimator {k1, k2, k3}``
     - KL estimator: ``k1`` for standard PPO penalty, ``k2`` ≈ ``k1`` when used as loss, ``k3``
       for GRPO loss.

Recommended pairings (the trainer warns if you mix them differently):

- **KL as reward penalty** (no ``--algo.kl.use_loss``): ``--algo.kl.estimator k1`` is the only
  sensible choice. Typical: ``--algo.kl.init_coef 0.01 --algo.kl.estimator k1`` for standard PPO
  or REINFORCE++.
- **KL as a loss term** (``--algo.kl.use_loss``): use ``--algo.kl.estimator k2`` or ``k3`` (k1 is
  not a valid loss). Typical: GRPO uses ``--algo.kl.use_loss --algo.kl.estimator k3``; the
  default RLVR recipe in :doc:`hybrid_engine` uses
  ``--algo.kl.use_loss --algo.kl.estimator k2 --algo.kl.init_coef 1e-5`` with
  REINFORCE++-baseline.

Entropy
~~~~~~~

- ``--actor.entropy_coef``: entropy regularization coefficient. ``None`` disables it; ``0`` only
  logs entropy without applying it as a loss.

Reward shaping
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--reward.normalize_enable``
     - Online running mean/std normalization of raw rewards.
   * - ``--reward.clip_range <low> <high>``
     - Clamp raw rewards before advantage computation (default ``-10 10``).
   * - ``--reward.overlong_buffer_len <L>``
     - DAPO-style soft length limit: penalize responses whose length exceeds
       ``max_new_tokens - L``.
   * - ``--reward.overlong_penalty_factor``
     - Multiplicative magnitude of the overlong penalty (default ``1.0``).
   * - ``--reward.stop_properly_penalty_coef <c>``
     - ProRL-style truncation penalty for samples with ``finish_reason='length'``.
       ``c >= 0`` scales the reward by ``c ∈ [0, 1]``; ``c < 0`` overrides the reward
       (e.g., ``-0.5``).

Off-policy correction (TIS / ICEPOP / Seq-Mask-TIS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because vLLM uses different kernels (and sometimes a different precision) than the trainer, the
same token sequence can produce slightly different log-probs in rollout vs. training. OpenRLHF can
apply **importance-sampling correction** to compensate.

Enable with ``--algo.advantage.is_correction_enable`` and pick a strategy:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Strategy
     - Flag
     - Behavior
   * - **TIS** *(default)*
     - ``--algo.advantage.is_correction_type tis``
     - Token-level **clamp** of the IS ratio into ``[low, high]``.
   * - **ICEPOP**
     - ``--algo.advantage.is_correction_type icepop``
     - Token-level **filter** — zero out coefficients outside ``[low, high]`` (no clamp).
   * - **Seq-Mask-TIS**
     - ``--algo.advantage.is_correction_type seq-mask-tis``
     - Sequence-level geometric-mean masking.

Thresholds: ``--algo.advantage.is_correction_threshold <low> <high>`` (default ``0.5 5.0``).
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

Dynamic sampling (DAPO)
~~~~~~~~~~~~~~~~~~~~~~~

For reasoning tasks, generate **multiple completions per prompt** and train only on a subset:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--rollout.n_samples_per_prompt``
     - Completions per prompt (must be > 1 for filtering / RLOO / GRPO / REINFORCE++-baseline).
   * - ``--algo.dynamic_filtering_enable``
     - Enable DAPO-style filtering by ``scores`` returned from the reward / agent function.
   * - ``--algo.dynamic_filtering_range <low> <high>``
     - Reward range to keep, e.g., ``0.0 1.0``. Samples outside the range are dropped.
   * - ``--rollout.vllm_generate_batch_size``
     - vLLM generation batch size; can exceed ``--rollout.batch_size`` for oversampling.
       Requires ``--train.async_enable`` when greater than ``--rollout.batch_size``.
   * - ``--train.dynamic_batch_enable``
     - Form micro-batches by token budget instead of count — much better packing for
       variable-length sequences. Pair with ``--train.max_tokens_per_gpu`` and
       ``--rollout.max_tokens_per_gpu``.

Sizing rule of thumb: ``train.batch_size = rollout.batch_size * rollout.n_samples_per_prompt``.
With ``--algo.dynamic_filtering_enable`` the effective batch may shrink if many samples are
filtered out — keep oversample headroom via ``--rollout.vllm_generate_batch_size`` (async only).

End-to-end DAPO recipe: `train_dapo_ray_hybrid_engine.sh
<https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_dapo_ray_hybrid_engine.sh>`_.

Sampling & misc
~~~~~~~~~~~~~~~

- ``--rollout.top_p`` / ``--rollout.temperature``: vLLM sampling parameters during rollouts.
- ``--critic.freezing_steps``: keep the actor frozen for the first *N* updates while the critic
  warms up.
- ``--critic.save_value_network``: also save the critic checkpoint (PPO only).
- ``--train.full_determinism_enable``: bit-reproducible behavior (slower; vLLM v1 + fixed seed
  paths).

Vision-Language Model (VLM) RLHF
--------------------------------

Since OpenRLHF 0.10, VLMs (e.g., **Qwen3.5**) can be trained end-to-end with image inputs. The
framework auto-detects VLMs via the ``vision_config`` field in the HuggingFace config, loads them
with ``AutoModelForImageTextToText``, uses ``AutoProcessor`` for correct image-token insertion, and
forwards images to vLLM for multimodal generation.

Why this matters: previous VLM RLHF setups required custom data loaders and bespoke inference
paths. OpenRLHF reuses the **same agent execution pipeline**, the **same RL algorithms**, and the
**same Hybrid Engine** for VLMs as for text-only models — you only add a few flags.

VLM-specific flags:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Flag
     - Meaning
   * - ``--data.image_key``
     - Dataset JSON key holding the image paths/URLs (default ``images``).
   * - ``--data.max_images_per_prompt``
     - Max images per prompt for vLLM (``0`` = text-only; default ``0``).
   * - ``--actor.freeze_visual_encoder``
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
      --actor.model_name_or_path Qwen/Qwen3.5-2B \
      --reward.remote_url examples/python/math_reward_func.py \
      --data.prompt_dataset hiyouga/geometry3k \
      --data.input_key prompt \
      --data.label_key label \
      --data.image_key images \
      --data.max_images_per_prompt 1 \
      --actor.freeze_visual_encoder \
      --data.max_len 4096 \
      --rollout.max_new_tokens 2048 \
      --algo.advantage.estimator reinforce_baseline \
      --train.colocate_all \
      --vllm.gpu_memory_utilization 0.7 \
      --data.apply_chat_template \
      --ds.attn_implementation eager \
      --ds.param_dtype bf16 \
      ...

.. note::

   - **Tested**: Qwen3.5 (hybrid linear + full attention).
   - **Auto-detected, not yet tested**: VLMs with a ``ForConditionalGeneration`` architecture
     (Gemma4, LLaVA, InternVL, ...).
   - Use ``--ds.attn_implementation eager`` for models with linear attention layers — flash
     attention may not support packed sequences there.
   - VLM training does **not** support ``--ds.packing_samples`` (packing collapses the batch
     dimension, breaking image-token alignment) or the PPO critic (use a critic-free
     ``--algo.advantage.estimator`` like ``reinforce_baseline`` / ``rloo`` / ``group_norm``).
   - Multi-turn VLM RLHF is supported; see `vlm_multiturn_agent.py
     <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/vlm_multiturn_agent.py>`_.

Logging & evaluation
--------------------

- ``--logger.wandb.key {token}`` / ``--logger.wandb.project`` / ``--logger.wandb.group`` /
  ``--logger.wandb.run_name``: Wandb logging.
- ``--logger.tensorboard_dir {logdir}``: TensorBoard logging.
- ``--logger.logging_steps``: log every *N* training steps.
- ``--eval.steps`` / ``--eval.dataset``: periodic evaluation on a held-out dataset.
- ``--train.num_episodes``: total RL episodes to run (one episode = one full rollout pass through
  ``rollout.batch_size`` prompts).

Training metrics include policy loss, KL, entropy, reward / advantage statistics, generation
length, grad norm, and per-phase wall-clock time. See :doc:`checkpoint` for save/resume mechanics
and :doc:`performance` for tuning.

Reference scripts
-----------------

- REINFORCE++-baseline + Hybrid Engine — `train_reinforce_baseline_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_hybrid_engine.sh>`_
- Async agent RLHF + Partial Rollout — `train_reinforce_baseline_ray_agent_async.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_reinforce_baseline_ray_agent_async.sh>`_
- DAPO with dynamic filtering — `train_dapo_ray_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_dapo_ray_hybrid_engine.sh>`_
- ProRL V2 (1.5B reasoning) — `train_prorlv2_math_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_prorlv2_math_hybrid_engine.sh>`_
- Custom Python reward (RFT) — `train_ppo_with_reward_fn.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_ppo_with_reward_fn.sh>`_
- VLM math RLHF — `train_vlm_math_hybrid_engine.sh
  <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_vlm_math_hybrid_engine.sh>`_

Design Paradigm: Agent-Based Execution
======================================

OpenRLHF is the first RLHF framework to **decouple the execution mode from the RL algorithm** through a unified, agent-based pipeline. Every training run — standard PPO, GRPO reasoning, multi-turn tool use, VLM math, or async DAPO — flows through the **same** ``AgentExecutorBase`` interface and produces the **same** kind of token-level trajectory. The RL loss layer treats all of them identically.

This page explains the design philosophy, why it matters in practice, and how the moving parts fit together. For runnable recipes see :doc:`agent_training`; for the underlying distributed plumbing see :doc:`architecture`.

.. contents::
   :local:
   :depth: 2

Why agent-based?
----------------

The traditional RLHF pipeline conflates two concerns: *how an experience is collected* (single-turn generation, multi-step interaction with an env, tool calls) and *how the policy is updated* (PPO, GRPO, REINFORCE++, ...). When these are entangled, every new execution mode requires changes deep in the trainer, and every new algorithm has to be re-implemented per mode.

OpenRLHF's agent paradigm separates the two:

- **Agent executors** produce a **token-level trajectory** — input tokens, output tokens, log-probs, optional environment images, per-step rewards.
- **RL algorithms** consume that trajectory through a single shared loss-computation layer.

Because the boundary is **token-in / token-out**, there is **zero** text-level re-tokenization or string concatenation between rollout and training, which is the most common source of subtle RLHF bugs (e.g., chat-template drift, BOS/EOS mismatches, special-token handling).

Agent architecture
------------------

.. code-block:: text

                 ┌─────────────────────────────┐
                 │    AgentExecutorBase        │
                 │  (Token-in-Token-out Core)  │
                 └─────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
         SingleTurnExecutor        MultiTurnExecutor
                 │                         │
      ┌──────────┴──────────┐   ┌─────────┴──────────┐
      ↓                     ↓   ↓                    ↓
  Standard RLHF      Custom Reward   Multi-Step    External Env
  (One-shot gen)     Function (RFT)  Reasoning     (OpenAI Agent Server)
      ↓                     ↓           ↓                ↓
      └─────────────────────┴───────────┴────────────────┘
                              │
                    Consistent Token Trajectories
                              │
                    ┌─────────┴─────────┐
                    │  RL Algorithms    │
                    │  (Decoupled)      │
                    │                   │
                    │  PPO, REINFORCE++ │
                    │  GRPO, RLOO, etc. │
                    └───────────────────┘

The dataflow in one sentence: **prompts → agent executor → token trajectories + rewards → advantage computation → policy / critic update**.

Core design principles
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Principle
     - What it means in practice
   * - **Token-in-Token-out**
     - Trajectories are stored as token IDs and log-probs, never re-detokenized between sampling and training. Eliminates text-level mismatches; exactly preserves whatever vLLM produced.
   * - **Unified interface**
     - Single-turn and multi-turn share the same ``AgentExecutorBase`` API. Switching modes is a one-flag change (``--agent_func_path``); the trainer code path is identical.
   * - **Algorithm-agnostic**
     - Algorithms (PPO, REINFORCE++, GRPO, RLOO, Dr. GRPO, REINFORCE++-baseline) are selected with ``--advantage_estimator`` and consume the same trajectory format. Adding a new algorithm doesn't require touching the executors.
   * - **Pipeline-agnostic**
     - Sync (Hybrid Engine) and async (with optional partial rollout) pipelines both feed the same loss layer. Throughput vs. on-policy-ness is a deployment choice, not an algorithmic one.
   * - **Extensible**
     - Custom reward (single-turn RFT) is a Python file path; custom environment (multi-turn) is an ``AgentInstanceBase`` subclass. No trainer modifications needed.
   * - **Production-ready**
     - Built-in support for resumable checkpoints, best-checkpoint tracking, EMA, multi-node SLURM, Wandb/TensorBoard, off-policy correction, length penalties, dynamic filtering — all modular flags on top of the same pipeline.

Execution modes (orthogonal to algorithms)
------------------------------------------

Execution mode and RL algorithm are two **independent** axes — every combination is valid.

.. list-table::
   :header-rows: 1
   :widths: 22 40 23 15

   * - Mode
     - Use cases
     - How to select
     - Complexity
   * - **Single-Turn** *(default)*
     - Standard RLHF, RLVR, Reinforced Fine-Tuning with custom rewards
       (math / code / format / multi-objective)
     - default; optionally ``--remote_rm_url`` (HTTP RM server or local Python file)
     - ⭐ — covers ~99% of use cases
   * - **Multi-Turn**
     - Multi-step reasoning with environment feedback, code execution loops, game playing,
       tool use, agent training
     - ``--agent_func_path /path/to/agent.py``
     - ⭐⭐ — implement ``reset()`` / ``step()``

Pipelines (sync vs. async)
--------------------------

A third orthogonal axis — choose based on your throughput / convergence trade-off.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Pipeline
     - Behavior
     - When to use
   * - **Sync (Hybrid Engine)**
     - Rollout → train → rollout (one phase at a time, models share GPUs via sleep mode)
     - Default. Best convergence stability; best throughput on colocated GPUs. See
       :doc:`hybrid_engine`.
   * - **Async**
     - Rollout and train run concurrently through a bounded queue (``--async_queue_size``);
       samples are slightly off-policy
     - Throughput-critical, convergence already validated. ``--async_train``.
   * - **Async + Partial Rollout**
     - Generation never stops — vLLM pause/resume swaps weights mid-flight
     - Maximum overlap; in-flight samples may contain old + new weight tokens.
       ``--async_train --partial_rollout``. See :doc:`async_training`.

Worked combinations
-------------------

A few combinations from the upstream reference scripts:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Algorithm × Mode × Pipeline
     - Example use
     - Reference script
   * - **PPO + Single-Turn + Sync**
     - Classical RLHF with a trained reward model
     - ``train_ppo_ray_hybrid_engine.sh``
   * - **REINFORCE++-baseline + Single-Turn + Sync**
     - RLVR / reasoning with rule-based rewards
     - ``train_prorlv2_math_hybrid_engine.sh``
   * - **REINFORCE++-baseline + Multi-Turn + Async + Partial Rollout**
     - Async agent RLHF with environment feedback
     - ``train_reinforce_baseline_ray_agent_async.sh``
   * - **REINFORCE++-baseline + VLM Single-Turn + Sync**
     - Vision-language math reasoning
     - ``train_vlm_math_hybrid_engine.sh``
   * - **PPO + Multi-Turn (OpenAI server) + Sync**
     - Tool-use training with an OpenAI-compatible chat API
     - ``agent_func_openai_server_executor.py``

Mental model in one sentence
----------------------------

**Pick three independent things — algorithm, execution mode, pipeline — and OpenRLHF runs the same training loop end-to-end.**

How to use
----------

**Selecting the execution mode**

- Single-turn (default): no extra flag, or ``--remote_rm_url <http-url-or-py-file>`` for custom rewards.
- Multi-turn: ``--agent_func_path /path/to/agent_func.py``.

**Selecting the RL algorithm** — set ``--advantage_estimator`` to one of:

- ``gae`` — PPO (default; uses a critic).
- ``reinforce`` — REINFORCE++.
- ``reinforce_baseline`` — REINFORCE++-baseline (recommended for RLVR / reasoning).
- ``rloo`` — RLOO.
- ``group_norm`` — GRPO.
- ``dr_grpo`` — Dr. GRPO.

**Selecting the pipeline**

- Sync + Hybrid Engine (default): ``--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep``.
- Async: ``--async_train`` (optionally ``--async_queue_size N``).
- Async + partial rollout: ``--async_train --partial_rollout``.

For runnable recipes covering every combination, see :doc:`agent_training`.

.. warning::
   When implementing a custom multi-turn agent, always follow the **token-in-token-out principle** — operate on token IDs returned by ``states["action_text"]`` etc., not on text strings concatenated by hand. Otherwise you may introduce silent text-level mismatches between sampling and training.

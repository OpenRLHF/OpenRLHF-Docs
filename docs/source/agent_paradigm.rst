Design Paradigm: Agent-Based Execution
======================================

OpenRLHF unifies generation and training under a single **token-in-token-out** agent pipeline. Every training run — standard PPO, GRPO, or a multi-turn reasoning loop — goes through the same ``AgentExecutorBase`` interface. This eliminates text-level mismatches between sampling and training and lets you switch execution modes without touching the RL algorithm.

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
  (One-shot gen)     Function      Reasoning     (OpenAI Agent Server)
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

Core principles
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Principle
     - What it means
   * - **Token-in-Token-out**
     - All sampling produces token-level trajectories — no text re-tokenization between generation and training.
   * - **Unified interface**
     - Single-turn and multi-turn share the same ``AgentExecutorBase`` API; switch modes with one flag.
   * - **Algorithm-agnostic**
     - RL algorithms (PPO, REINFORCE++, GRPO, RLOO, ...) are decoupled from execution mode — any algorithm works with any mode.
   * - **Extensible**
     - Plug in custom reward functions (single-turn) or environments (multi-turn) via a Python file.

Two execution modes (orthogonal to algorithms)
----------------------------------------------

Execution mode and RL algorithm are independent axes. Any combination is valid.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Mode
     - Use cases
     - How to select
   * - **Single-Turn** (default)
     - Standard RLHF, RLVR, custom reward functions
     - Default; optionally ``--remote_rm_url``
   * - **Multi-Turn**
     - Multi-step reasoning, interactive environments, tool use
     - ``--agent_func_path /path/to/agent.py``

Algorithm is selected via ``--advantage_estimator`` (``gae`` / ``reinforce`` / ``reinforce_baseline`` / ``rloo`` / ``group_norm`` / ``dr_grpo``). See :doc:`agent_training` for the full recipe catalog, the algorithm table, and code templates for both modes.

.. warning::
   When implementing custom agents, always follow the **token-in-token-out principle** to keep sampling and training consistent.

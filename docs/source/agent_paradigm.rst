Design Paradigm: Agent-Based Execution
=======================================

**On top of the Ray distributed architecture**, OpenRLHF is **the first RLHF framework** to implement a **unified agent-based paradigm**. Every training run—whether standard PPO or complex multi-turn reasoning—follows a consistent agent execution pipeline.

Why Agent-Based?
----------------

OpenRLHF **unifies generation and training through token-in-token-out agent execution**, ensuring perfect consistency, easy single/multi-turn extension, and zero text-level mismatches.

Agent Architecture
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
  (One-shot gen)     Function      Reasoning     (NeMo Gym)
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

Core Design Principles
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Principle
     - Description
     - Benefit
   * - **Token-in-Token-out**
     - All sampling produces token-level trajectories
     - Zero text-level mismatch
   * - **Unified Interface**
     - Same ``AgentExecutorBase`` API for all modes
     - Switch modes with one flag
   * - **Algorithm-Agnostic**
     - RL algorithms (PPO, REINFORCE++, etc.) are decoupled from agent executors
     - Any algorithm works with any mode
   * - **Extensible**
     - Plug in custom rewards/environments easily
     - Rapid experimentation
   * - **Production-Ready**
     - Sync/Async/Hybrid Engine support
     - From research to deployment

Two Execution Modes (Orthogonal to Algorithms)
-----------------------------------------------

**Important**: Execution modes are **completely independent** from RL algorithms. You can use **any algorithm** (PPO, REINFORCE++, GRPO, RLOO) with **any execution mode** (single-turn or multi-turn).

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Mode
     - Use Cases
     - Configuration
     - Complexity
   * - **Single-Turn**
     - Standard RLHF, custom reward functions
     - Default or ``--remote_rm_url``
     - ⭐ Default
   * - **Multi-Turn**
     - Multi-step reasoning, interactive environments
     - ``--agent_func_path``
     - ⭐⭐ Advanced

**Example Combinations** (all valid):
- PPO + Single-Turn
- PPO + Multi-Turn
- REINFORCE++ + Single-Turn
- REINFORCE++ + Multi-Turn
- GRPO + Single-Turn
- GRPO + Multi-Turn
- (Any algorithm + Any mode)

See :doc:`agent_training` for the consolidated training + execution guide (single-turn and multi-turn).

Key Advantages
--------------

1. **Consistency**: Token-level trajectories ensure perfect alignment between generation and training
2. **Flexibility**: Switch between single-turn and multi-turn with a single flag
3. **Algorithm Independence**: Use any RL algorithm with any execution mode
4. **Extensibility**: Easy to implement custom rewards and environments
5. **Production Ready**: Supports synchronous, asynchronous, and hybrid engine modes

How to Use
----------

**Selecting Execution Mode**:

- **Single-Turn** (default): No extra flag needed, or use ``--remote_rm_url`` for custom rewards
- **Multi-Turn**: Set ``--agent_func_path /path/to/agent.py``

**Selecting RL Algorithm**:

Use ``--advantage_estimator`` flag:

- ``gae`` (default): PPO
- ``reinforce``: REINFORCE++
- ``reinforce_baseline``: REINFORCE++-baseline  
- ``rloo``: RLOO
- ``group_norm``: GRPO
- ``dr_grpo``: Dr. GRPO

.. warning::
   When implementing custom agents, always follow the **token-in-token-out principle** to ensure consistency between sampling and training.


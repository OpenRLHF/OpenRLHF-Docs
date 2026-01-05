Welcome to OpenRLHF's documentation!
===================================

OpenRLHF is a high-performance, production-ready RLHF framework combining **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm**.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

What Makes OpenRLHF Unique?
----------------------------

**Core Innovation**: Unified agent-based paradigm that decouples execution modes from RL algorithms

- **🎯 Agent-Based Paradigm**: Token-in-token-out execution with two orthogonal dimensions (see :doc:`agent_paradigm`)
  
  - **Execution Modes**: :doc:`single_turn_agent` (default) | :doc:`multi_turn_agent` (advanced)
  - **RL Algorithms**: Any algorithm works with any mode (see :doc:`rl`)

- **🏗️ Distributed Architecture**: Ray + vLLM for scalable training up to 70B+ parameters (see :doc:`architecture`)
- **⚡ Production Ready**: Hybrid Engine, async training, comprehensive monitoring (see :doc:`hybrid_engine` and :doc:`performance`)

Quick Links
-----------

- **Getting Started**: :doc:`quick_start` - Installation and first RLHF training
- **Core Concepts**: :doc:`architecture` (Ray + vLLM) | :doc:`agent_paradigm` (Design Paradigm)
- **Agent Execution**: :doc:`single_turn_agent` (default, 99% use cases) | :doc:`multi_turn_agent` (advanced)
- **Training Methods**: :doc:`rl` (PPO, REINFORCE++, GRPO) | :doc:`non_rl` (DPO, KTO, etc.)
- **Advanced Topics**: :doc:`hybrid_engine` | :doc:`performance` | :doc:`multi-node`

Resources
---------

- `GitHub Repository <https://github.com/OpenRLHF/OpenRLHF>`_
- `Technical Report <https://www.researchgate.net/publication/393414548_OpenRLHF_An_Easy-to-use_Scalable_and_High-performance_RLHF_Framework>`_
- `Slides <https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing>`_

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :caption: Core Architecture

   architecture
   agent_paradigm

.. toctree::
   :maxdepth: 2
   :caption: Agent Execution Modes

   single_turn_agent
   multi_turn_agent

.. toctree::
   :maxdepth: 2
   :caption: Training Methods

   rl
   non_rl

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   hybrid_engine
   performance
   multi-node
   checkpoint
   sequence_parallelism

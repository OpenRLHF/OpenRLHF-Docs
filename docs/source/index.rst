Welcome to OpenRLHF's documentation!
====================================

OpenRLHF is a high-performance, production-ready RLHF framework combining **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm**.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

Start here
----------

New to OpenRLHF? Start with :doc:`quick_start` (includes a short reading guide). Use :doc:`troubleshooting` when you hit issues.

What makes OpenRLHF unique?
---------------------------

**Core Innovation**: Unified agent-based paradigm that decouples execution modes from RL algorithms

- **Agent-Based Paradigm**: Token-in-token-out execution with two orthogonal dimensions (see :doc:`agent_paradigm`)
  
  - **Execution Modes**: :ref:`single_turn_mode` (default) | :ref:`multi_turn_mode` (advanced) in :doc:`agent_training`
  - **Training Guide**: Recipes + algorithms + modes in one place (see :doc:`agent_training`)

- **Distributed Architecture**: Ray + vLLM for scalable training up to 70B+ parameters (see :doc:`architecture`)
- **Production Ready**: Hybrid Engine, async training, comprehensive monitoring (see :doc:`hybrid_engine` and :doc:`performance`)

Quick Links
-----------

- **Getting Started**: :doc:`quick_start` - Installation and first RLHF training
- **Core Concepts**: :doc:`architecture` (Ray + vLLM) | :doc:`agent_paradigm` (Design Paradigm)
- **Agent Execution**: :ref:`single_turn_mode` (default) | :ref:`multi_turn_mode` (advanced) in :doc:`agent_training`
- **Training Guide**: :doc:`agent_training` (SFT/RM/RL + single/multi-turn) | :doc:`non_rl` (DPO, KTO, etc.)
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
   :caption: Getting Started (Read in order)

   quick_start
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   agent_paradigm
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Training Guides

   agent_training
   non_rl

.. toctree::
   :maxdepth: 2
   :caption: Scaling & Operations

   hybrid_engine
   performance
   checkpoint
   sequence_parallelism
   multi-node
   nvidia_docker

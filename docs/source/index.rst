Welcome to OpenRLHF's documentation!
===================================

`OpenRLHF <https://github.com/OpenRLHF/OpenRLHF>`_ is **the first** high-performance, production-ready open-source RLHF framework that combines **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm** for scalable and extensible reinforcement learning from human feedback.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

Key Innovations
---------------

**🏗️ Ray + vLLM Distributed Architecture**

OpenRLHF is **the first RLHF framework** built on Ray + vLLM distributed architecture, orchestrating multiple components across GPUs efficiently:

- **Ray**: Distributed scheduler separating Actor, Reward, Reference, and Critic models across GPUs, enabling scalable training for 70B+ parameters
- **vLLM**: High-performance inference engine with AutoTP/PP—RLHF training spends 80% time on generation, vLLM optimizes this critical path
- **Hybrid Engine**: All models and vLLM engines share GPU resources, minimizing idle time and maximizing utilization
- **DeepSpeed**: ZeRO-3, deepcompile, AutoTP, and RingAttention for memory-efficient training

**🎯 Unified Agent-Based Design Paradigm**

OpenRLHF is **the first RLHF framework** to implement a **unified agent-based paradigm**. Every training run—whether standard PPO or complex multi-turn reasoning—follows a consistent agent execution pipeline:

- **Token-in-Token-out**: Perfect consistency between generation and training, zero text-level mismatches
- **Single-Turn Mode** (Default): One-shot generation, covers 99% use cases including standard RLHF and custom reward functions
- **Multi-Turn Mode** (Advanced): Multi-step interactions with environment feedback for reasoning chains, coding, game playing
- **Algorithm-Agnostic**: All RL algorithms (PPO, REINFORCE++, GRPO, RLOO) work with both execution modes
- **Extensible**: Easy to plug in custom rewards, environments, and agent logic

**🚀 State-of-the-Art RL Algorithms**

- **PPO**: Full critic network, stable and proven
- **REINFORCE++**: PPO tricks without critic, efficient and memory-friendly
- **REINFORCE++-baseline**: Best for reasoning tasks (RLVR), robust to reward scales
- **RLOO**: Per-token KL + PPO-clip for multi-sample training
- **GRPO**: Group normalization for batch-based training
- **Dr. GRPO**: Simplified GRPO variant

All algorithms are **decoupled from execution modes**—use any algorithm with single-turn or multi-turn agent execution.

Features
--------

- **Production-Ready**: From research to deployment with sync/async/hybrid engine modes
- **Scalability**: Train models up to 70B+ parameters efficiently
- **Efficiency**: 80% generation time optimized with vLLM, sample packing, dynamic batching
- **Flexibility**: LoRA/QLoRA, MoE, FlashAttention, RingAttention, multi-node SLURM
- **Extensibility**: Custom reward functions, external environments (NeMo Gym)
- **Monitoring**: Wandb, TensorBoard, comprehensive logging

For more technical details, see our `slides <https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing>`_ and `technical report <https://www.researchgate.net/publication/393414548_OpenRLHF_An_Easy-to-use_Scalable_and_High-performance_RLHF_Framework>`_.

Getting Started
---------------

Check out the :doc:`quick_start` section for installation and typical workflow.

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
   :caption: Training Methods

   rl
   non_rl

.. toctree::
   :maxdepth: 2
   :caption: Agent Execution Modes

   single_turn_agent
   multi_turn_agent

.. toctree::
   :maxdepth: 2
   :caption: Advanced Features

   hybrid_engine
   performance
   multi-node
   checkpoint
   sequence_parallelism

Welcome to OpenRLHF's documentation!
====================================

OpenRLHF is a high-performance, production-ready RLHF framework that combines a **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm** for scalable and extensible reinforcement learning from human feedback.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

Core features
-------------

- **Ray + vLLM distributed architecture** — scales to 70B+ parameter models with vLLM-accelerated generation, DeepSpeed ZeRO-3 training, and Ray-based scheduling.
- **Unified agent-based paradigm** — token-in-token-out pipeline that decouples execution mode (single-turn / multi-turn) from RL algorithm. Any algorithm pairs with any mode.
- **State-of-the-art RL algorithms** — PPO, REINFORCE++, REINFORCE++-baseline, GRPO, Dr. GRPO, RLOO.
- **Hybrid Engine** — colocate Actor / Critic / Reward / Reference / vLLM on the same GPUs with sleep-mode memory sharing for maximum utilization.
- **Async training & Partial Rollout** — overlap rollout with training; partial rollout overlaps weight sync with generation via vLLM pause/resume.
- **Single-turn rewards & multi-turn agents** — HTTP remote RM, custom Python reward functions (RFT), or full multi-turn environments with optional OpenAI-compatible chat server.
- **Vision-Language Model RLHF** *(new in 0.10)* — train VLMs (e.g., Qwen3.5) end-to-end with image inputs.
- **Off-policy correction** — TIS / ICEPOP / Seq-Mask-TIS to handle vLLM↔training log-prob mismatches.
- **Production essentials** — resumable checkpoints, best-checkpoint tracking, EMA, Wandb / TensorBoard logging, SLURM multi-node, LoRA / QLoRA (non-RL trainers).

Why OpenRLHF
------------

- **Performance** — vLLM-accelerated generation eliminates the 80% rollout bottleneck; Hybrid Engine eliminates GPU idle time.
- **Flexibility** — switch between sync / async pipelines, single / multi-turn modes, and any of six RL algorithms with single-flag changes.
- **Compatibility** — native HuggingFace model loading; no custom checkpoint format.
- **Production-ready** — full checkpoint/resume, best-model tracking, multi-node SLURM, comprehensive logging.
- **Extensible** — plug in custom reward functions or multi-turn environments via a Python file; no trainer modifications needed.

Start here
----------

- **New users**: :doc:`quick_start` — installation and your first training run.
- **Mental model**: :doc:`architecture` and :doc:`agent_paradigm`.
- **Pick a recipe**: :doc:`agent_training` (RL) or :doc:`non_rl` (SFT / RM / DPO).
- **Something broke**: :doc:`troubleshooting`.

Resources
---------

- `GitHub Repository <https://github.com/OpenRLHF/OpenRLHF>`_
- `Technical Report <https://www.researchgate.net/publication/393414548_OpenRLHF_An_Easy-to-use_Scalable_and_High-performance_RLHF_Framework>`_
- `Slides <https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing>`_
- `vLLM blog: Accelerating RLHF with vLLM <https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html>`_

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
   :caption: Core Concepts

   architecture
   agent_paradigm
   hybrid_engine
   async_training

.. toctree::
   :maxdepth: 2
   :caption: Training Guides

   agent_training
   non_rl
   common_options

.. toctree::
   :maxdepth: 2
   :caption: Scaling & Operations

   performance
   checkpoint
   sequence_parallelism
   multi-node
   nvidia_docker
   troubleshooting

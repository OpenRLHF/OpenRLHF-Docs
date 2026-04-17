Welcome to OpenRLHF's documentation!
====================================

OpenRLHF is a high-performance, production-ready RLHF framework that combines a **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm** for scalable and extensible reinforcement learning from human feedback.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

Highlights
----------

- **Ray + vLLM distributed architecture** — scales to 70B+ models. vLLM-accelerated generation
  eliminates the dominant RLHF bottleneck; DeepSpeed ZeRO-3 trains directly from HuggingFace
  checkpoints with no model conversion.
- **Unified agent-based paradigm** — token-in-token-out pipeline that decouples *execution mode*
  (single-turn / multi-turn) from *RL algorithm*. Any algorithm pairs with any mode through a
  single shared loss layer.
- **State-of-the-art RL algorithms** — PPO, REINFORCE++, REINFORCE++-baseline, GRPO, Dr. GRPO,
  RLOO; switchable with one flag.
- **Hybrid Engine** — colocate Actor / Critic / Reward / Reference / vLLM on the **same** GPUs
  with sleep-mode memory sharing. Highest utilization on small clusters; simplest deployment.
- **Async training & Partial Rollout** — overlap rollout with training, and overlap weight sync
  with generation via vLLM pause / resume. Highest throughput when convergence is validated.
- **Single-turn rewards & multi-turn agents** — HTTP remote RM, custom Python reward functions
  (Reinforced Fine-Tuning), full multi-turn environments, or wrap vLLM as an OpenAI-compatible
  chat server.
- **Vision-Language Model RLHF** *(new in 0.10)* — train VLMs (e.g., Qwen3.5) end-to-end with
  image inputs through the same agent pipeline.
- **Off-policy correction** — TIS / ICEPOP / Seq-Mask-TIS handle vLLM ↔ training log-prob
  mismatches.
- **Production essentials** — resumable checkpoints, best-checkpoint tracking, EMA, Wandb /
  TensorBoard logging, SLURM multi-node, LoRA / QLoRA for SFT / RM / DPO.

Start here
----------

.. list-table::
   :widths: 25 75

   * - **New users**
     - :doc:`quick_start` — install + first training run.
   * - **Mental model**
     - :doc:`architecture` (components) and :doc:`agent_paradigm` (design).
   * - **Pick a recipe**
     - :doc:`agent_training` (RL training) or :doc:`non_rl` (SFT / RM / DPO).
   * - **Look up a flag**
     - :doc:`common_options` (shared) or the trainer-specific page above.
   * - **Scale or tune**
     - :doc:`hybrid_engine`, :doc:`async_training`, :doc:`performance`, :doc:`multi-node`.
   * - **Something broke**
     - :doc:`troubleshooting`.

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

.. toctree::
   :maxdepth: 2
   :caption: Training Guides

   agent_training
   non_rl
   common_options

.. toctree::
   :maxdepth: 2
   :caption: Scaling & Operations

   hybrid_engine
   async_training
   performance
   checkpoint
   sequence_parallelism
   multi-node
   nvidia_docker
   troubleshooting

Welcome to OpenRLHF's documentation!
====================================

OpenRLHF is a high-performance, production-ready RLHF framework that combines a **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm** for scalable and extensible reinforcement learning from human feedback.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture
   :align: center
   :width: 700px

What's in OpenRLHF
------------------

**Distributed architecture (Ray + vLLM + DeepSpeed)**
   Scales to 70B+ parameter models. vLLM accelerates generation; DeepSpeed ZeRO-3 + AutoTP + RingAttention shard training; Ray orchestrates everything across GPUs and nodes. See :doc:`architecture`.

**Unified agent-based execution paradigm**
   Token-in-token-out pipeline that decouples *execution mode* (single-turn / multi-turn) from *RL algorithm* (PPO, REINFORCE++, GRPO, RLOO, ...). Any algorithm pairs with any mode. See :doc:`agent_paradigm`.

**State-of-the-art RL algorithms**
   PPO, REINFORCE++, REINFORCE++-baseline, GRPO, Dr. GRPO, RLOO — selectable with one flag (``--advantage_estimator``). See the algorithm table in :doc:`agent_training`.

**Hybrid Engine + async + partial rollout**
   ``--colocate_all_models`` colocates Actor / Critic / Reward / Reference / vLLM on the same GPUs with sleep-mode memory sharing for maximum utilization. ``--async_train`` overlaps rollout with training; ``--partial_rollout`` overlaps weight sync with generation via vLLM pause/resume. See :doc:`hybrid_engine` and :doc:`performance`.

**Vision-Language Model (VLM) RLHF** *(new in 0.10)*
   Train VLMs (e.g., Qwen3.5) end-to-end with image inputs. Auto-detection via ``vision_config``, ``AutoProcessor`` for multimodal token insertion, multi-image prompts, and optional vision-encoder freezing. See the VLM section in :doc:`agent_training`.

**Single-turn rewards & multi-turn agents**
   Single-turn: HTTP remote RM (``--remote_rm_url``) or local Python reward function (RFT). Multi-turn: ``--agent_func_path`` plugs in a custom environment with ``reset()`` / ``step()`` (or wrap an OpenAI-compatible chat server). See :doc:`agent_training`.

**Off-policy correction**
   TIS / ICEPOP / Seq-Mask-TIS for vLLM rollout↔training log-prob mismatch (``--enable_vllm_is_correction`` + ``--vllm_is_correction_type``).

**Production essentials**
   Resumable checkpoints (``--save_steps`` / ``--load_checkpoint``), best-checkpoint tracking (``--best_metric_key``), EMA (``--enable_ema``), Wandb / TensorBoard logging, SLURM multi-node, LoRA / QLoRA for non-RL trainers.

Start here
----------

- New users: :doc:`quick_start` — installation and your first training run.
- Mental model: :doc:`agent_paradigm` and :doc:`architecture`.
- Something broke: :doc:`troubleshooting`.

Quick Links
-----------

- **Getting Started**: :doc:`quick_start`
- **Core Concepts**: :doc:`agent_paradigm` | :doc:`architecture`
- **Training Guides**: :doc:`agent_training` (RM, RL, single/multi-turn, VLM) | :doc:`non_rl` (SFT, DPO) | :doc:`common_options`
- **Scaling & Ops**: :doc:`hybrid_engine` | :doc:`performance` | :doc:`checkpoint` | :doc:`sequence_parallelism` | :doc:`multi-node`

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

   agent_paradigm
   architecture

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
   performance
   checkpoint
   sequence_parallelism
   multi-node
   nvidia_docker
   troubleshooting

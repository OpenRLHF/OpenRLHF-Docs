Architecture: Ray + vLLM
========================

OpenRLHF orchestrates RLHF training across GPUs using a few well-established components:

Ray — distributed scheduler
---------------------------

`Ray <https://github.com/ray-project/ray>`_ schedules the Actor, Reward, Reference, and Critic models (and vLLM engines) across GPUs. Models can be placed on separate GPUs for large-scale training, or colocated on shared GPUs via the :doc:`hybrid_engine` to maximize utilization. This enables RLHF at up to **70B+ parameters**.

vLLM — inference engine
-----------------------

RL training spends most of its wall-clock time on sample generation. `vLLM <https://github.com/vllm-project/vllm>`_ with Auto Tensor Parallelism (AutoTP) and Pipeline Parallelism (PP) provides high-throughput, memory-efficient generation.

DeepSpeed — memory-efficient training
-------------------------------------

Training uses `DeepSpeed <https://github.com/deepspeedai/DeepSpeed>`_ ZeRO-3 with optional `deepcompile <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md>`_, `AutoTP <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md>`_, and RingAttention — so you can train large models directly from HuggingFace checkpoints without a heavyweight custom framework.

Transformers & NCCL
-------------------

Models are loaded through HuggingFace Transformers (no conversion step). Inter-GPU communication uses NCCL / CUDA IPC for weight sync and collective ops.

See :doc:`agent_paradigm` for how these pieces are tied together into a unified training pipeline.

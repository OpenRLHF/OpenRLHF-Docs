Architecture Foundation: Ray + vLLM Distribution
=================================================

OpenRLHF is **the first RLHF framework** built on Ray + vLLM distributed architecture, orchestrating multiple components across GPUs efficiently.

Core Infrastructure Components
-------------------------------

Ray - Distributed Scheduler and Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenRLHF leverages `Ray <https://github.com/ray-project/ray>`_ for efficient distributed scheduling. It separates the Actor, Reward, Reference, and Critic models across different GPUs, enabling scalable training for models up to **70B+ parameters**.

**Hybrid Engine Scheduling**: All models and vLLM engines can share GPU resources—minimizing idle time and maximizing GPU utilization. This allows running full RLHF pipelines on limited hardware.

vLLM - High-Performance Inference Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLHF training spends **80% of the time on sample generation**. Powered by `vLLM <https://github.com/vllm-project/vllm>`_ with Auto Tensor Parallelism (AutoTP) and Pipeline Parallelism (PP), OpenRLHF delivers high-throughput, memory-efficient generation.

DeepSpeed - Memory-Efficient Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Built on `DeepSpeed <https://github.com/deepspeedai/DeepSpeed>`_ ZeRO-3, `deepcompile <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md>`_, `AutoTP <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md>`_, and RingAttention. Enables large model training without heavyweight frameworks while working directly with HuggingFace models.

Transformers - Model Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native integration with HuggingFace Transformers for seamless model loading, state management, and fine-tuning of pretrained models.

NCCL / CUDA IPC - High-Speed Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient inter-GPU communication for distributed training and inference.

Key Benefits
------------

- **Scalability**: Train models up to 70B+ parameters efficiently
- **Efficiency**: 80% generation time optimized with vLLM
- **Flexibility**: Hybrid Engine shares GPUs to avoid resource idling
- **Compatibility**: Native HuggingFace model integration
- **Performance**: High-speed NCCL communication


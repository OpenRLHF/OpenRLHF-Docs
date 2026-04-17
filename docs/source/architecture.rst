Architecture Foundation: Ray + vLLM Distribution
=================================================

OpenRLHF is built on a **Ray + vLLM distributed architecture**, orchestrating multiple components across GPUs efficiently. The framework cleanly separates concerns: Ray handles distributed scheduling, vLLM handles high-throughput generation, and DeepSpeed handles memory-efficient training, all working with native HuggingFace models.

.. image:: _static/openrlhf-arch.png
   :alt: OpenRLHF Architecture (Ray + vLLM)
   :align: center
   :width: 700px

Why this matters
----------------

RLHF training spends roughly **80% of wall-clock time on sample generation**. A naive setup would dedicate separate GPUs to generation and training, leaving each idle for most of the time. OpenRLHF avoids this in two ways:

- **Distributed mode**: Actor / Reward / Reference / Critic models and vLLM engines are placed on different GPUs and pipelined. This scales to **70B+ parameter** models.
- **Hybrid Engine mode** (recommended for most workloads): all models and vLLM engines colocate on the **same** GPUs and use sleep mode to free memory between phases (see :doc:`hybrid_engine`).

Core infrastructure components
------------------------------

Ray — Distributed scheduler and controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenRLHF uses `Ray <https://github.com/ray-project/ray>`_ for distributed scheduling. It separates the Actor, Reward, Reference, and Critic models across different GPUs (or colocates them via Hybrid Engine), enabling scalable training for models up to **70B+ parameters**. Ray placement groups make it easy to specify per-model GPU resources, and Ray jobs let you submit a training run from any node in the cluster.

vLLM — High-performance inference engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`vLLM <https://github.com/vllm-project/vllm>`_ provides high-throughput, memory-efficient generation through PagedAttention, continuous batching, prefix caching, and CUDA graphs. OpenRLHF integrates vLLM with **Auto Tensor Parallelism (AutoTP)** and **Pipeline Parallelism (PP)**, so a single rollout can be sharded across multiple GPUs without changing the training loop.

Weight sync between the trainer and vLLM uses NCCL (``--vllm_sync_backend nccl``) for low-latency updates after each training step.

DeepSpeed — Memory-efficient training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training uses `DeepSpeed <https://github.com/deepspeedai/DeepSpeed>`_ ZeRO-3 to shard optimizer state, gradients, and parameters across GPUs. Optional features:

- `deepcompile <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md>`_ — graph compilation (``--deepcompile``).
- `AutoTP <https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md>`_ — DeepSpeed tensor parallelism (``--ds_tensor_parallel_size``).
- `RingAttention <https://arxiv.org/abs/2310.01889>`_ for long contexts (``--ring_attn_size``; see :doc:`sequence_parallelism`).

The result: large-model training works directly from HuggingFace checkpoints — no model conversion, no custom training framework.

Transformers — Model interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native integration with HuggingFace `Transformers <https://github.com/huggingface/transformers>`_ for model loading, state management, and fine-tuning. VLMs (Vision-Language Models) are auto-detected via the ``vision_config`` field and loaded with ``AutoModelForImageTextToText`` (see :doc:`agent_training`).

NCCL / CUDA IPC — High-speed communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inter-GPU communication uses NCCL for collective ops and CUDA IPC for intra-node weight transfer. This keeps the cost of weight sync, gradient reduce, and KV-cache movement low even at 70B scale.

Key benefits
------------

- **Scalability**: train models up to 70B+ parameters efficiently.
- **Efficiency**: vLLM-accelerated generation eliminates the dominant RLHF bottleneck.
- **Flexibility**: Hybrid Engine shares GPUs to avoid resource idling on small clusters; distributed mode scales out for large models.
- **Compatibility**: native HuggingFace model integration — no custom checkpoint format.
- **Production-ready**: NCCL-backed weight sync, async pipelines, partial rollout, and full checkpoint/resume.

See :doc:`agent_paradigm` for how these pieces are tied together into the unified training pipeline, and :doc:`hybrid_engine` / :doc:`performance` for tuning.

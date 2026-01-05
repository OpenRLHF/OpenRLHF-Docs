Performance Tuning Guide
===================================

Overview
--------

OpenRLHF's agent-based architecture provides multiple execution modes with different performance characteristics. This guide helps you optimize for your hardware and workload.

**Key Principle**: Maximize GPU utilization while maintaining training stability.

See :doc:`agent_paradigm` for architecture overview and :doc:`hybrid_engine` for Hybrid Engine details.

Execution Mode Selection
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Mode
     - When to Use
     - Configuration
   * - **Hybrid Engine (Default)**
     - Sufficient GPU memory
     - ``--colocate_all_models`` + ``--vllm_enable_sleep`` + ``--deepspeed_enable_sleep``
   * - **Asynchronous**
     - Throughput critical, convergence validated
     - ``--async_train`` + ``--agent_func_path``

Resource Allocation (Distributed Mode)
---------------------------------------

**Recommended Ratio**: ``vLLM : Actor : Critic = 1:1:1``

Example: 70B model on 48 A100 GPUs

- 16 GPUs → vLLM Engine
- 16 GPUs → Actor Model  
- 16 GPUs → Critic Model

This ensures balanced utilization across generation and training phases.

Speed Optimizations
-------------------

High Priority (Always Enable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Optimization
     - Flag
     - Benefit
   * - **Sample Packing**
     - ``--packing_samples``
     - 2-3x training speedup
   * - **vLLM NCCL Backend**
     - ``--vllm_sync_backend nccl``
     - Faster weight sync
   * - **Dynamic Batch**
     - ``--use_dynamic_batch``
     - Better GPU utilization

Medium Priority (When Memory Allows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Optimization
     - Flag
     - Requirement
   * - **Hybrid Engine**
     - ``--colocate_all_models`` + ``--vllm_enable_sleep`` + ``--deepspeed_enable_sleep``
     - Sufficient GPU memory
   * - **Overlap Comm**
     - ``--overlap_comm``
     - Sufficient GPU memory
   * - **DeepCompile**
     - ``--deepcompile``
     - PyTorch 2.0+
   * - **Prefix Caching**
     - vLLM config
     - ``n_samples_per_prompt`` > 1

Low Priority (For Throughput)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Optimization
     - Flag
     - Note
   * - **Async Training**
     - ``--async_train``
     - May affect stability

Memory Management
-----------------

When You Have Enough Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Enable**:

- Disable ``--adam_offload``
- Enable ``--overlap_comm``
- Use ``--colocate_critic_reward`` and ``--colocate_actor_ref``
- Use ``--colocate_all_models`` (Hybrid Engine)

When Hitting OOM
~~~~~~~~~~~~~~~~

❌ **Disable**:

- All ``--colocate_*`` options
- ``--overlap_comm``

✅ **Enable**:

- ``--adam_offload``
- ``--gradient_checkpointing``
- Increase ``--zero_stage`` (2 → 3)

✅ **Reduce**:

- ``--micro_train_batch_size``
- ``--micro_rollout_batch_size``
- vLLM Tensor Parallel size

Batch Size Tuning
-----------------

Generation Phase
~~~~~~~~~~~~~~~~

**Goal**: Maximize throughput

- ✅ Maximize ``--micro_rollout_batch_size``
- ✅ Minimize vLLM TP size (use more engines instead)
- ✅ Use ``--enable_prefix_caching`` when ``n_samples_per_prompt`` > 1

Training Phase
~~~~~~~~~~~~~~

**Goal**: Maximize GPU utilization

- ✅ Maximize ``--micro_train_batch_size``
- ✅ Enable ``--packing_samples``
- ✅ Use ``--use_dynamic_batch`` for variable sequence lengths


Quick Start Templates
---------------------

8x A100 (80GB) - Hybrid Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 8 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 8 \
      --vllm_num_engines 4 \
      --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.5 \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep \
      --vllm_sync_backend nccl \
      --packing_samples \
      --use_dynamic_batch \
      ... # other training args

16x A100 (80GB) - Distributed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 4 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 4 \
      --critic_num_nodes 1 \
      --critic_num_gpus_per_node 4 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 \
      --vllm_tensor_parallel_size 2 \
      --vllm_sync_backend nccl \
      --packing_samples \
      ... # other training args

SFT/RM/DPO Training
-------------------

For supervised training tasks:

- ✅ Always enable ``--packing_samples``
- ✅ Use ``--gradient_checkpointing`` when memory constrained
- ✅ Use ``--zero_stage 2`` or ``3`` based on model size
- ✅ Enable ``--overlap_comm`` when memory allows

Batch Inference
---------------

For batch inference with OpenRLHF:

- ✅ Enable `prefix caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ when ``best_of_n`` > 1
- ✅ Use vLLM's batched inference mode
- ✅ Adjust ``--vllm_gpu_memory_utilization`` for throughput

Monitoring and Debugging
-------------------------

Performance Metrics
~~~~~~~~~~~~~~~~~~~

Monitor these metrics during training:

- **Generation throughput**: tokens/second
- **Training throughput**: samples/second  
- **GPU utilization**: should be >80%
- **Memory usage**: should not hit OOM

Common Issues
~~~~~~~~~~~~~

**Low GPU Utilization**

- ✅ Increase batch sizes
- ✅ Enable Hybrid Engine
- ✅ Reduce vLLM TP size

**OOM Errors**

- ✅ Reduce batch sizes
- ✅ Disable ``--colocate_*`` options
- ✅ Enable ``--adam_offload``
- ✅ Increase ``--zero_stage``

**Slow Generation**

- ✅ Use ``--vllm_sync_backend nccl``
- ✅ Reduce vLLM TP size
- ✅ Enable prefix caching

**Training Divergence (Async Mode)**

- ✅ Reduce ``OPENRLHF_ASYNC_QUEUE_SIZE``
- ✅ Switch to synchronous or Hybrid Engine mode
- ✅ Adjust learning rate

Advanced Optimizations
-----------------------

For Large Models (70B+)
~~~~~~~~~~~~~~~~~~~~~~~

- Use higher ``--zero_stage`` (3)
- Increase vLLM Tensor Parallel size
- Use Pipeline Parallelism for vLLM
- Enable ``--gradient_checkpointing``
- Consider model quantization (``--load_in_4bit``)

For Long Context (>8K tokens)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Enable RingAttention (``--ring_attn_size``)
- Use ``--packing_samples`` aggressively
- Adjust ``--max_len`` carefully
- Monitor memory usage closely

For Multi-Node Training
~~~~~~~~~~~~~~~~~~~~~~~~

- Use SLURM scripts for job management
- Ensure high-speed interconnect (InfiniBand)
- Use ``--vllm_sync_backend nccl``
- Monitor network utilization

Summary
-------

**Default Setup** (Good for most cases)

.. code-block:: bash

   --packing_samples \
   --vllm_sync_backend nccl \
   --gradient_checkpointing

**High Performance** (When memory allows)

.. code-block:: bash

   --colocate_all_models \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --packing_samples \
   --vllm_sync_backend nccl \
   --overlap_comm

**Maximum Throughput** (May affect stability)

.. code-block:: bash

   --async_train \
   --agent_func_path /path/to/agent.py \
   --packing_samples \
   --vllm_sync_backend nccl \
   --use_dynamic_batch

**Memory Constrained**

.. code-block:: bash

   --adam_offload \
   --gradient_checkpointing \
   --zero_stage 3 \
   --packing_samples

See :doc:`agent_paradigm` for execution mode details and :doc:`hybrid_engine` for Hybrid Engine configuration.

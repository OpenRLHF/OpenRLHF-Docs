Hybrid Engine
=============

Overview
--------

The **Hybrid Engine** allows all models (Actor, Critic, Reward, Reference) and vLLM engines to **share GPUs**, maximizing resource utilization and avoiding idle time during different phases of training.

**Key Benefits**:

- **Maximum GPU utilization**: No idle GPUs during generation or training
- **Memory efficiency**: Share memory between models
- **Performance**: Better throughput than distributed mode
- **Simplicity**: Fewer GPUs needed for full RLHF pipeline

See :doc:`agent_paradigm` for architecture overview and :doc:`performance` for tuning guide.

.. _hybrid_engine:

Ray RLHF using Hybrid Engine
-----------------------------

OpenRLHF supports the hybrid engine, allowing all models and vLLM engines to share the GPUs to avoid resource idling.

.. code-block:: bash
   
   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # if you want to launch ray on more nodes, use
   ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 \
      --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 \
      --reward_num_gpus_per_node 8 \
      --actor_num_nodes 1 \
      --actor_num_gpus_per_node 8 \
      --vllm_num_engines 8 \
      --vllm_tensor_parallel_size 1 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.6 \
      --advantage_estimator reinforce_baseline \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 4 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 8 \
      --rollout_batch_size 1024 \
      --n_samples_per_prompt 4 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --max_samples 20000 \
      --generate_max_len 1024 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 1e-4 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --normalize_reward \
      --gradient_checkpointing \
      --packing_samples \
      --vllm_sync_backend nccl \
      --enforce_eager \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep

.. note::
   - **Agent Execution**: Works with both single-turn and multi-turn modes (see :doc:`agent_training`)
   - **Algorithm Compatibility**: Works with all RL algorithms (change ``--advantage_estimator`` to switch)

Options
-------

Core Options

- ``--colocate_all_models``: Colocate vLLM Engines, Actor, Reference, Reward and Critic Model nodes (Hybrid Engine)
- ``--vllm_gpu_memory_utilization``: vLLM gpu_memory_utilization (larger value means more memory usage and better performance). Recommended: ``0.5`` for 8x A100
- ``--vllm_enable_sleep``: Enable sleep mode for vLLM when using ``--colocate_all_models``
- ``--deepspeed_enable_sleep``: Enable sleep mode for DeepSpeed engines when using ``--colocate_all_models``
- ``--enforce_eager``: Disable CUDA graph for vLLM (required for some setups)

Additional Options

- ``--colocate_critic_reward``: Colocate Critic and Reward models (use when not using ``--colocate_all_models``)
- ``--colocate_actor_ref``: Colocate Actor and Reference models (use when not using ``--colocate_all_models``)
- ``--ref_reward_offload``: Offload Reference and Reward models to CPU during training

When to Use Hybrid Engine
--------------------------

✅ **Use Hybrid Engine when**:

- You have sufficient GPU memory
- You want maximum GPU utilization
- You have 8+ GPUs available
- Need simplest possible setup

❌ **Don't use Hybrid Engine when**:

- Hitting OOM errors
- Training very large models with limited memory

Memory Requirements
-------------------

**Rule of Thumb**: ``vllm_gpu_memory_utilization`` + Model Memory < 1.0

Example for 8x A100 (80GB):

- 8B model: ``--vllm_gpu_memory_utilization 0.6``
- 13B model: ``--vllm_gpu_memory_utilization 0.5``
- 34B model: ``--vllm_gpu_memory_utilization 0.4``
- 70B model: Use more GPUs or distributed mode

Performance Tips
----------------

1. **Start Conservative**: Begin with ``--vllm_gpu_memory_utilization 0.5``
2. **Monitor Memory**: Watch for OOM errors
3. **Increase Gradually**: Increase to 0.6 or 0.7 if stable
4. **Enable Sleep Modes**: Always use ``--vllm_enable_sleep`` and ``--deepspeed_enable_sleep``
5. **Use NCCL**: Set ``--vllm_sync_backend nccl`` for faster weight sync

Async training (mutually exclusive with Hybrid Engine)
------------------------------------------------------

Async training overlaps rollout and training to increase throughput, but it may affect stability (more off-policy).

**Important**: Async training and Hybrid Engine are **mutually exclusive**. Do **not** use ``--async_train`` together with ``--colocate_all_models``.

Minimal async setup
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export VLLM_USE_V1=1
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --agent_func_path /path/to/agent_func.py \
      --async_train \
      --async_queue_size 1 \
      ... # other training args

Key knobs
~~~~~~~~~

- ``--async_train``: enable async training
- ``--async_queue_size``: async buffer size (larger may be more off-policy, default 1)

Troubleshooting
---------------

**OOM Errors**

- ✅ Reduce ``--vllm_gpu_memory_utilization`` (0.5 → 0.4)
- ✅ Reduce batch sizes
- ✅ Disable ``--colocate_all_models`` and use distributed mode
- ✅ Enable ``--packing_samples``


**Low Throughput**

- ✅ Increase ``--vllm_gpu_memory_utilization`` (0.5 → 0.6)
- ✅ Increase batch sizes

**vLLM Hangs**

- ✅ Use ``--enforce_eager``
- ✅ Set ``--vllm_sync_backend nccl``
- ✅ Check CUDA/NCCL versions

See :doc:`performance` for comprehensive tuning guide.
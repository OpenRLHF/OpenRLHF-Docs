Hybrid Engine
=============

The **Hybrid Engine** colocates all models (Actor, Critic, Reward, Reference) and vLLM engines on the **same** GPUs so none sit idle between the generation and training phases. It is the recommended mode when GPU memory allows — typically the best throughput and the simplest setup.

See :doc:`agent_paradigm` for the overall architecture and :doc:`performance` for the broader tuning guide.

.. _hybrid_engine:

Launch recipe
-------------

.. code-block:: bash

   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # additional worker nodes (optional)
   ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --ref_num_nodes 1 --ref_num_gpus_per_node 8 \
      --reward_num_nodes 1 --reward_num_gpus_per_node 8 \
      --actor_num_nodes 1 --actor_num_gpus_per_node 8 \
      --vllm_num_engines 8 --vllm_tensor_parallel_size 1 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.6 \
      --advantage_estimator reinforce_baseline \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 4 --train_batch_size 128 \
      --micro_rollout_batch_size 8 --rollout_batch_size 1024 \
      --n_samples_per_prompt 4 --max_epochs 1 \
      --max_len 2048 --max_samples 20000 \
      --zero_stage 3 --param_dtype bf16 \
      --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
      --init_kl_coef 1e-4 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages --apply_chat_template \
      --normalize_reward --gradient_checkpointing \
      --packing_samples --use_dynamic_batch \
      --vllm_sync_backend nccl --enforce_eager \
      --vllm_enable_sleep --deepspeed_enable_sleep

.. note::
   Works with both single-turn and multi-turn agent modes (see :doc:`agent_training`) and with any RL algorithm (change ``--advantage_estimator`` to switch).

Key flags
---------

Hybrid-engine essentials:

- ``--colocate_all_models``: colocate vLLM engines, Actor, Reference, Reward, and Critic on the same GPUs.
- ``--vllm_gpu_memory_utilization <0..1>``: vLLM KV-cache fraction. Start at ``0.5`` for 8×A100 and raise if stable.
- ``--vllm_enable_sleep`` / ``--deepspeed_enable_sleep``: sleep-mode for vLLM / DeepSpeed when colocated — frees memory between phases.
- ``--enforce_eager``: disable CUDA graphs for vLLM (required for some setups).
- ``--vllm_sync_backend nccl``: NCCL backend for weight sync (faster than the default).

Finer-grained colocation (when **not** using ``--colocate_all_models``):

- ``--colocate_critic_reward``: place Critic and Reward on the same GPUs.
- ``--colocate_actor_ref``: place Actor and Reference on the same GPUs.
- ``--ref_reward_offload``: offload Reference and Reward to CPU during training.

Rule of thumb: ``vllm_gpu_memory_utilization`` + model memory < 1.0. For 8×A100 (80GB): ~0.6 for 8B, ~0.5 for 13B, ~0.4 for 34B. For 70B+, prefer distributed mode.

Async training is mutually exclusive
------------------------------------

Do **not** combine ``--async_train`` with ``--colocate_all_models``. Async pipelines overlap rollout and training for higher throughput at the cost of more off-policy samples; see the async section in :doc:`agent_training` for configuration.

See :doc:`performance` for the full tuning guide and :doc:`troubleshooting` for OOM / NCCL / vLLM-hang issues.

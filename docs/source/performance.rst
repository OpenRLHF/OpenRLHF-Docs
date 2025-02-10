Performance Tuning Guide
===================================

Ray PPO
-----------

To achieve optimal performance, we recommend allocating nodes `vLLM:Actor:Critic = 1:1:1`. 

- For example, for a 70B model with 48 A100 GPUs, it is advised to allocate 16 A100 GPUs to the vLLM Engine, 16 GPUs to the Actor model, and the remaining 16 GPUs to the Critic model. 
- Enable the ``--colocate_critic_reward``, ``--colocate_actor_ref`` options to merge nodes.  
- You should increase the ``rollout_micro_batch_size`` (and minimize the TP size of vLLM engine) as much as possible. During the training phase, a larger ``--micro_train_batch_size`` is better and enable ``--packing_samples``.
- When there are enough GPUs, please disable ``--adam_offload`` and enable ``--overlap_comm``.
- For multi-nodes RLHF, please use ``--vllm_sync_backend nccl`` with vLLM 0.7.2+.
- Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation when ``n_samples_per_prompts`` > 1.
- Using hybrid engine ``--colocate_all_models`` and ``--vllm_enable_sleep`` rather than distributed RLHF when the model size and context length are small values.

SFT/RM/DPO/PPO training
------------------

- Enable ``--packing_samples`` in the training scripts


Batch Inference
---------------

- Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation when ``best_of_n`` > 1.

Performance Tuning Guide
===================================

Ray PPO
-----------

To achieve optimal performance, we recommend allocating more nodes to the vLLM Engine. 

- For example, for a 70B model with 32 A100 GPUs, it is advised to allocate more than 16 A100 GPUs to the vLLM Engine, 8 GPUs to the Actor model, and the remaining 8 GPUs to the Critic model. 
- Enable the ``--colocate_critic_reward``, ``--colocate_actor_ref``, and ``--ref_reward_offload`` options to merge nodes.  
- You should increase the `rollout_micro_batch_size` (and minimize the TP size of vLLM engine) as much as possible, and avoid `Reward/Reference` models forward OOM (Out Of Memory) issues using `--packing_samples`. During the training phase, a larger `--micro_train_batch_size` is better.
- Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation when ``n_samples_per_prompt`` > 1.

Batch Inference
---------------

- Enable `enable_prefix_caching <https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html>`_ in vLLM generation when ``best_of_n`` > 1.
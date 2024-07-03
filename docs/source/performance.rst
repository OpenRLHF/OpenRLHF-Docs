Performance Tuning (PPO)
===================================

To achieve optimal performance, we recommend allocating more nodes to the vLLM Engine. 
For example, for a 70B model with 32 A100 GPUs, it is advised to allocate more than 16 A100 GPUs to the vLLM Engine, 8 GPUs to the Actor model, and the remaining 8 GPUs to the Critic model. 
Additionally, enable the ``--colocate_critic_reward``, ``--colocate_actor_ref``, and ``--ref_reward_offload`` options to merge nodes.  
Finally, you should increase the micro-batch-size (and minimize the TP size of vLLM engine) as much as possible while avoiding OOM (Out Of Memory) issues, especially during the generation phase of PPO.
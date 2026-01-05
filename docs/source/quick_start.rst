Quick Start
===========

.. _installation:

Installation
------------

To use OpenRLHF, first launch the docker container (**Recommended**) and ``pip install`` openrlhf inside the docker container:

.. code-block:: bash

   # Launch the docker container
   docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.02-py3 bash
   pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y

   # pip install
   pip install openrlhf

   # If you want to use vLLM acceleration (To install vLLM 0.13.0 - recommended)
   pip install openrlhf[vllm]
   # latest vLLM is also supported
   pip install openrlhf[vllm_latest]
   # Install vLLM, ring-flash-attention and Liger-Kernel
   pip install openrlhf[vllm,ring,liger]

   # pip install the latest version
   pip install git+https://github.com/OpenRLHF/OpenRLHF.git

   # Or git clone
   git clone https://github.com/OpenRLHF/OpenRLHF.git
   cd OpenRLHF
   pip install -e .

.. note:: We recommend using vLLM 0.13.0+ for best performance. 
   We also provided the `Dockerfiles for vLLM <https://github.com/OpenRLHF/OpenRLHF/tree/main/dockerfile>`_ and :ref:`nvidia-docker`.

Understanding OpenRLHF Architecture
-----------------------------------

Before diving into training, it's helpful to understand OpenRLHF's architecture:

**🏗️ Ray + vLLM Distributed Architecture**

OpenRLHF leverages Ray for distributed scheduling and vLLM for high-performance generation. See :doc:`architecture` for details.

**🎯 Unified Agent-Based Paradigm**

All training runs through a consistent agent execution pipeline with two independent dimensions:

- **Execution Modes**: Single-Turn (default) or Multi-Turn (advanced)
- **RL Algorithms**: PPO, REINFORCE++, GRPO, RLOO (via ``--advantage_estimator``)

**Key Point**: These two dimensions are **completely decoupled**—any algorithm works with any mode.

See :doc:`agent_paradigm` for comprehensive overview.

Prepare Datasets
----------------

OpenRLHF provides flexible data processing methods:

**Key Parameters**:

- ``--input_key``: Specify JSON key name for input data
- ``--apply_chat_template``: Use HuggingFace tokenizer's `chat template <https://huggingface.co/docs/transformers/main/en/chat_templating>`_
- ``--input_template``: Custom template string (alternative to chat template)
- ``--prompt_data_probs`` / ``--dataset_probs``: Mix multiple datasets (e.g., ``0.1,0.4,0.5``)
- ``--eval_dataset``: Specify evaluation dataset path

**Chat Template Example**:

.. code-block:: python
      
   dataset = [{"input_key": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
      {"role": "user", "content": "I'd like to show off how chat templating works!"},
   ]}]

   tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)
   # Output: "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"

.. note:: 
   JSON key options vary by dataset type. See:
   
   - `Reward Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10>`_
   - `SFT Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9>`_
   - `Prompt Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6>`_

Pretrained Models
-----------------

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using:

- ``--pretrain``: Actor model
- ``--reward_pretrain``: Reward model
- ``--critic_pretrain``: Critic model

We have provided some pre-trained checkpoints and datasets on `HuggingFace OpenRLHF <https://huggingface.co/OpenRLHF>`_.

Typical RLHF Workflow
---------------------

1. **Supervised Fine-tuning (SFT)**

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_sft \
      --max_len 4096 \
      --dataset Open-Orca/OpenOrca \
      --input_key question \
      --output_key response \
      --input_template $'User: {}\nAssistant: ' \
      --train_batch_size 256 \
      --micro_train_batch_size 2 \
      --max_samples 500000 \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --save_path ./checkpoint/llama3-8b-sft \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --zero_stage 2 \
      --max_epochs 1 \
      --packing_samples \
      --bf16 \
      --learning_rate 5e-6 \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

See :doc:`rl` for detailed options.

2. **Reward Model Training**

.. code-block:: bash

   deepspeed --module openrlhf.cli.train_rm \
      --save_path ./checkpoint/llama3-8b-rm \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --train_batch_size 256 \
      --micro_train_batch_size 1 \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --bf16 \
      --max_epochs 1 \
      --max_len 8192 \
      --zero_stage 3 \
      --learning_rate 9e-6 \
      --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
      --apply_chat_template \
      --chosen_key chosen \
      --rejected_key rejected \
      --packing_samples \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

See :doc:`rl` for detailed options.

3. **RL Training with Agent Execution (Single-Turn Mode)**

All RL training uses the **unified agent execution pipeline**. The following example shows **single-turn agent mode** (default, 99% use cases):

.. code-block:: bash

   # launch the master node of ray in container
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

   # if you want to launch ray on more nodes, use
   ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

   # Run RL training with Hybrid Engine (Recommended)
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
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 8 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 16 \
      --rollout_batch_size 1024 \
      --n_samples_per_prompt 1 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --max_samples 100000 \
      --generate_max_len 1024 \
      --zero_stage 3 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data OpenRLHF/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --normalize_reward \
      --gradient_checkpointing \
      --packing_samples \
      --vllm_sync_backend nccl \
      --enforce_eager \
      --vllm_enable_sleep \
      --deepspeed_enable_sleep \
      --use_wandb {wandb_token}

.. note::
   **Agent Execution**: This uses **single-turn agent mode** (default). Switch algorithms via ``--advantage_estimator`` (see :doc:`agent_paradigm` for all options).

.. tip::
   **For reasoning tasks**: Use ``--advantage_estimator reinforce_baseline`` (REINFORCE++-baseline). See :doc:`rl` for algorithm details.

.. tip::
   **Ray Environment Setup**: Let Ray auto-deploy with ``--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'``

.. note::
   **GPU Index Errors**: Set ``export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` if you encounter DeepSpeed GPU device setup issues.

Advanced Agent Modes
--------------------

**Single-Turn with Custom Rewards**

Use custom reward functions instead of a trained reward model:

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain meta-llama/Meta-Llama-3-8B \
      --remote_rm_url /path/to/reward_func.py \
      --label_key answer \
      ... # other training args

See :doc:`single_turn_agent` for details.

**Multi-Turn Agent for Complex Interactions**

For multi-step reasoning, coding with feedback, or game playing:

.. code-block:: bash

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --agent_func_path /path/to/agent.py \
      --async_train \
      ... # other training args

See :doc:`multi_turn_agent` for details.

Next Steps
----------

- **RL Training**: See :doc:`rl` for comprehensive RL training guide
- **Agent Modes**: See :doc:`single_turn_agent` and :doc:`multi_turn_agent` for advanced agent usage
- **Performance**: See :doc:`performance` for tuning guide
- **Hybrid Engine**: See :doc:`hybrid_engine` for maximum GPU utilization
- **Non-RL Methods**: See :doc:`non_rl` for DPO, KTO, rejection sampling
- **Multi-Node**: See :doc:`multi-node` for distributed training

More examples and scripts available in `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

.. _nvidia-docker:

One-Click Installation Script of Nvidia-Docker
-----------------------------------------------

.. code-block:: bash

   # remove old docker
   sudo apt-get autoremove docker docker-ce docker-engine docker.io containerd runc
   dpkg -l |grep ^rc|awk '{print $2}' |sudo xargs dpkg -P
   sudo apt-get autoremove docker-ce-*
   sudo rm -rf /etc/systemd/system/docker.service.d
   sudo rm -rf /var/lib/docker

   # install docker
   curl https://get.docker.com | sh \
   && sudo systemctl --now enable docker

   # install nvidia-docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
         && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
         && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
               sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
               sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker

   sudo groupadd docker
   sudo usermod -aG docker $USER
   newgrp docker
   docker ps

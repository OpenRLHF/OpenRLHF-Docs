Quick Start
=====

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

   # If you want to use vLLM acceleration (To install vLLM 0.9.1)
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

.. note:: We recommend using vLLM 0.9.1+. 
   We also provided the `Dockerfiles for vLLM <https://github.com/OpenRLHF/OpenRLHF/tree/main/dockerfile>`_  and  :ref:`nvidia-docker`.

Prepare Datasets
----------------

OpenRLHF provides multiple data processing methods in our dataset classes.
Such as in the `Prompt Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6>`_:

.. code-block:: python

   def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
      if apply_chat_template:
         chat = data[input_key]
         if isinstance(chat, str):
               chat = [{"role": "user", "content": chat}]
         prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      else:
         prompt = data[input_key]
         if input_template:
               prompt = input_template.format(prompt)
      return prompt

- We can use ``--input_key`` to specify the ``JSON key name`` of the input datasets ``--prompt_data {name or path}`` (PPO) or ``--dataset {name or path}``, and use ``--apply_chat_template`` to utilize the ``chat_template`` from the `Huggingface Tokenizer <https://huggingface.co/docs/transformers/main/en/chat_templating>`_.
- If you don't want to use ``--apply_chat_template``, you can use ``--input_template`` instead, or preprocess the datasets offline in advance.
- OpenRLHF also support mixing multiple datasets using ``--prompt_data_probs 0.1,0.4,0.5`` (PPO) or ``--dataset_probs 0.1,0.4,0.5``.

How Chat Templating Works:

.. code-block:: python
      
   dataset = [{"input_key": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
      {"role": "user", "content": "I'd like to show off how chat templating works!"},
   ]}]

   tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

   "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"

How to specify test datasets ?

Please set test datasets path using ``--eval_dataset {name or path}``.


.. note:: The ``JSON key`` options depends on the specific datasets. 
   See  `Reward Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10>`_ and `SFT Dataset <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9>`_

Pretrained Models
-----------------

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using ``--pretrain``, ``--reward_pretrain`` and ``--critic_pretrain``.
We have provided some pre-trained checkpoints and datasets on `HuggingFace OpenRLHF <https://huggingface.co/OpenRLHF>`_.

RLHF with Ray and vLLM
----------------
Then you can use the startup scripts we provide in the `examples <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples>`_ directory, or start the training using the following command using Hybrid Engine:


.. code-block:: bash

   python3 -m openrlhf.cli.train_ppo_ray \
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
      --advantage_estimator reinforce \
      --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
      --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
      --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
      --save_hf_ckpt \
      --micro_train_batch_size 4 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 8 \
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
      --init_kl_coef 1e-3 \
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


- For more details about the PPO and other online RLHF algorithms with Ray and vLLM, please refer to :ref:`rayppo`.
- OpenRLHF provides usage scripts and docs for the supported algorithms in `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_ and :doc:`usage`.

.. _nvidia-docker:

One-Click Installation Script of Nvidia-Docker
---------------------------

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

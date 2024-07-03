Quick Start
=====

.. _installation:

Installation
------------

To use OpenRLHF, first ``git clone`` it and launch the docker container (**Recommended**):

.. code-block:: bash

   git clone https://github.com/openllmai/OpenRLHF.git

   # If you need to use vLLM, please build a Docker image to avoid dependency issues (Optional)
   docker build -t nvcr.io/nvidia/pytorch:24.02-py3 ./OpenRLHF/dockerfile

   # Launch the docker container
   docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD/OpenRLHF:/openrlhf nvcr.io/nvidia/pytorch:24.02-py3 bash

Then ``pip install`` openrlhf inside the docker container

.. code-block:: bash

   cd /openrlhf
   pip install --user .

   cd examples


Prepare Datasets
----------------

We provide multiple data processing methods in our dataset classes.
Such as in the `Prompt Dataset <https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/prompts_dataset.py#L6>`_:

.. code-block:: python

   def preprocess_data(data, input_template=None, input_key=None, apply_chat_template=None) -> str:
      # custom dataset
      if input_key:
         if apply_chat_template:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
            input_template = None
         else:
            prompt = data[input_key]
      else:
         # Open-Orca/OpenOrca
         if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + " " + data["question"]
         .....

      # input template
      if input_template:
         prompt = input_template.format(prompt)
      return prompt

- We can use ``--input_key`` to specify the ``JSON key name`` of the input datasets ``--prompt_data {name or path}`` or ``--dataset {name or path}``, and use ``--apply_chat_template`` to utilize the ``chat_template`` from the `Huggingface Tokenizer <https://huggingface.co/docs/transformers/main/en/chat_templating>`_.
- If you don't want to use ``apply_chat_template``, you can use ``--input_template`` instead, or preprocess the data format in advance.
- We also support mixing multiple datasets using ``--prompt_data_probs 0.1,0.4,0.5`` or ``dataset_probs 0.1,0.4,0.5``.

Chat Templating Format

  .. code-block:: python
      
   dataset = [{"input_key": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
      {"role": "user", "content": "I'd like to show off how chat templating works!"},
   ]}]

   tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

   "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"

.. note:: The JSON key options depends on the specific datasets. 
   See  `Reward Dataset <https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/reward_dataset.py#L10>`_ and `SFT Dataset <https://github.com/OpenLLMAI/OpenRLHF/blob/7e436a673b9603847429971290cfd46029c4b52b/openrlhf/datasets/sft_dataset.py#L9>`_

Pretrained Models
-----------------

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can directly specify the model name or path using ``--pretrain``, ``--reward_pretrain`` and ``--critic_pretrain``.
We have provided some pre-trained checkpoints and datasets on `HuggingFace OpenLLMAI <https://huggingface.co/OpenLLMAI>`_.

PPO without Ray
----------------
Then you can use the startup scripts we provide in the `examples <https://github.com/OpenLLMAI/OpenRLHF/tree/main/examples>`_ directory, and start the training using the following command:


.. code-block:: bash

   deepspeed ./train_ppo.py \
      --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
      --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
      --save_path ./checkpoint/llama-3-8b-rlhf \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --micro_train_batch_size 2 \
      --train_batch_size 128 \
      --micro_rollout_batch_size 4 \
      --rollout_batch_size 1024 \
      --max_epochs 1 \
      --prompt_max_len 1024 \
      --generate_max_len 1024 \
      --zero_stage 2 \
      --bf16 \
      --actor_learning_rate 5e-7 \
      --critic_learning_rate 9e-6 \
      --init_kl_coef 0.01 \
      --prompt_data OpenLLMAI/prompt-collection-v0.1 \
      --input_key context_messages \
      --apply_chat_template \
      --max_samples 100000 \
      --normalize_reward \
      --adam_offload \
      --flash_attn \
      --gradient_checkpointing \
      --use_wandb {wandb_token}

- For the Ray PPO and vLLM, please refer to :ref:`rayppo`.
- We provide usage scripts and docs for the supported algorithms in `examples/scripts <https://github.com/OpenLLMAI/OpenRLHF/tree/main/examples/scripts>`_ and :doc:`usage`.


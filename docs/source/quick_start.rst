Quick Start
===========

Run first, then understand, then optimize.

- Install below, then jump to a recipe in :doc:`agent_training` or :doc:`non_rl`.
- Mental model: :doc:`agent_paradigm` and :doc:`architecture`.
- Tuning / scaling: :doc:`performance`, :doc:`hybrid_engine`, :doc:`multi-node`.
- Errors: :doc:`troubleshooting`.

.. _installation:

Installation
------------

The recommended path is to install inside an NVIDIA PyTorch container:

.. code-block:: bash

   docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
     -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.11-py3 bash

   # remove packages that conflict with vLLM / flash-attn in the base image
   pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y

   # pip install (pick one)
   pip install openrlhf                    # core only
   pip install openrlhf[vllm]              # + vLLM 0.19.0 (recommended)
   pip install openrlhf[vllm_latest]       # + vLLM > 0.19.0
   pip install openrlhf[vllm,ring,liger]   # + ring-flash-attention + Liger

   # or install from source
   git clone https://github.com/OpenRLHF/OpenRLHF.git
   cd OpenRLHF && pip install -e .

.. note::
   We recommend vLLM 0.19.0+ for best performance. See the `Dockerfiles <https://github.com/OpenRLHF/OpenRLHF/tree/main/dockerfile>`_ and :ref:`nvidia-docker`.

Prepare datasets
----------------

Key dataset flags (shared across trainers):

- ``--input_key``: JSON key holding the input (text or chat messages).
- ``--apply_chat_template``: use the tokenizer's `chat template <https://huggingface.co/docs/transformers/main/en/chat_templating>`_.
- ``--input_template``: custom template string (alternative to chat template).
- ``--dataset_probs`` / ``--prompt_data_probs``: mix multiple datasets (e.g., ``0.1,0.4,0.5``).
- ``--eval_dataset``: evaluation dataset path.

Chat-template example:

.. code-block:: python

   dataset = [{"input_key": [
       {"role": "user", "content": "Hello, how are you?"},
       {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
       {"role": "user", "content": "I'd like to show off how chat templating works!"},
   ]}]

   tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)
   # "<s>[INST] Hello, how are you? [/INST]I'm doing great. ...</s> [INST] ... [/INST]"

JSON key conventions vary by dataset type — see `Reward <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10>`_, `SFT <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9>`_, and `Prompt <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6>`_ dataset modules.

Pretrained models
-----------------

Checkpoints are HuggingFace-compatible. Point the trainer at them with:

- ``--pretrain`` — actor / base model
- ``--reward_pretrain`` — reward model
- ``--critic_pretrain`` — critic model (PPO)

Pre-trained examples are on `HuggingFace OpenRLHF <https://huggingface.co/OpenRLHF>`_.

Typical RLHF workflow
---------------------

1. **SFT** → :ref:`train_sft` in :doc:`non_rl`.
2. **Reward model** → :ref:`train_rm` in :doc:`agent_training`.
3. **RL training (Ray + vLLM)** → :ref:`rayppo` in :doc:`agent_training`.

Advanced modes (custom rewards, multi-turn envs, async, VLM) are all covered in :doc:`agent_training`. More runnable scripts live in `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

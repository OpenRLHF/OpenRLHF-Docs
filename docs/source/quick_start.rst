Quick Start
===========

This documentation is organized to help you **run first**, then **understand**, then **optimize/scale**.

How to read this documentation
------------------------------

- **Run my first training**: continue with the installation steps below.
- **Understand the mental model**: see :doc:`agent_paradigm` and :doc:`architecture`.
- **Training recipes + execution modes**: see :doc:`agent_training` and :doc:`non_rl`.
- **Performance / scaling**: see :doc:`performance`, :doc:`hybrid_engine`, and :doc:`multi-node`.
- **If you hit errors**: see :doc:`troubleshooting`.

.. _installation:

Installation
------------

To use OpenRLHF, first launch the docker container (**Recommended**) and ``pip install`` openrlhf inside the docker container:

.. code-block:: bash

   # Launch the docker container
   docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.11-py3 bash
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

Before you go deeper
--------------------

- For the mental model (execution modes are orthogonal to algorithms), see :doc:`agent_paradigm`.
- For how Ray + vLLM scale generation/training, see :doc:`architecture`.
- If you hit runtime issues (OOM, vLLM hang, GPU mapping), see :doc:`troubleshooting`.

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

Use this as a checklist, then jump to the canonical recipes:

1. **Supervised Fine-tuning (SFT)**: see :ref:`train_sft` in :doc:`non_rl`.
2. **Reward Model training (RM)**: see :ref:`train_rm` in :doc:`agent_training`.
3. **RL training (Ray + vLLM)**: see :ref:`rayppo` in :doc:`agent_training`.

Advanced Agent Modes
--------------------

- **Single-turn (remote RM / custom rewards)** and **Multi-turn (envs / async)**: see :doc:`agent_training`.

Next Steps
----------

- **Agent-based Training**: See :doc:`agent_training` for recipes + modes + algorithms
- **Performance**: See :doc:`performance` for tuning guide
- **Hybrid Engine**: See :doc:`hybrid_engine` for maximum GPU utilization
- **Non-RL Methods**: See :doc:`non_rl` for DPO, KTO, rejection sampling
- **Multi-Node**: See :doc:`multi-node` for distributed training

More examples and scripts available in `examples/scripts <https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

If you need NVIDIA Docker setup, see :ref:`nvidia-docker`.

Quick Start
===========

Run first, then understand, then optimize.

.. contents::
   :local:
   :depth: 1

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
   We recommend vLLM 0.19.0+ for best performance. See the `Dockerfiles
   <https://github.com/OpenRLHF/OpenRLHF/tree/main/dockerfile>`_ and :ref:`nvidia-docker`.

First run (RLVR with Qwen3-4B)
------------------------------

This is the canonical RL example used throughout the docs — REINFORCE++-baseline on math reasoning
with a Python reward function (no reward-model training required). Requires 4 GPUs:

.. code-block:: bash

   # 1) start ray on the head node
   ray start --head --node-ip-address 0.0.0.0 --num-gpus 4

   # 2) submit the training job
   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json='{"working_dir": "/openrlhf"}' \
      -- python3 -m openrlhf.cli.train_ppo_ray \
      --pretrain Qwen/Qwen3-4B-Thinking-2507 \
      --remote_rm_url examples/python/math_reward_func.py \
      --prompt_data zhuzilin/dapo-math-17k \
      --input_key prompt \
      --label_key label \
      --apply_chat_template \
      --packing_samples \
      --ref_num_nodes 1 --ref_num_gpus_per_node 4 \
      --actor_num_nodes 1 --actor_num_gpus_per_node 4 \
      --vllm_num_engines 2 --vllm_tensor_parallel_size 2 \
      --colocate_all_models \
      --vllm_gpu_memory_utilization 0.7 \
      --vllm_enable_sleep --deepspeed_enable_sleep \
      --vllm_sync_backend nccl --enforce_eager \
      --advantage_estimator reinforce_baseline \
      --use_kl_loss --kl_estimator k2 --init_kl_coef 1e-5 \
      --rollout_batch_size 128 --n_samples_per_prompt 8 \
      --train_batch_size 1024 \
      --max_len 8192 --max_new_tokens 4096 \
      --zero_stage 3 --param_dtype bf16 \
      --gradient_checkpointing \
      --actor_learning_rate 5e-7 \
      --save_path ./exp/Qwen3-4B-Thinking

For the full recipe (with off-policy correction, dynamic filtering, length budgets, ring attention,
checkpointing) see :doc:`hybrid_engine`. To swap in your own dataset, RM, or multi-turn
environment see :doc:`agent_training`.

Prepare datasets
----------------

Key dataset flags (shared across trainers):

- ``--input_key``: JSON key holding the input (text or chat messages).
- ``--apply_chat_template``: use the tokenizer's `chat template
  <https://huggingface.co/docs/transformers/main/en/chat_templating>`_.
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

JSON key conventions vary by dataset type — see `Reward
<https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10>`_,
`SFT <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9>`_,
and `Prompt
<https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6>`_
dataset modules.

Pretrained models
-----------------

Checkpoints are HuggingFace-compatible. Point the trainer at them with:

- ``--pretrain`` — actor / base model
- ``--reward_pretrain`` — reward model
- ``--critic_pretrain`` — critic model (PPO)

Pre-trained examples are on `HuggingFace OpenRLHF <https://huggingface.co/OpenRLHF>`_.

Typical RLHF workflow
---------------------

The full three-stage RLHF pipeline:

1. **SFT** → :ref:`train_sft` in :doc:`non_rl`.
2. **Reward model** → :ref:`train_rm` in :doc:`non_rl`.
3. **RL training** (Ray + vLLM) → :ref:`rayppo` in :doc:`agent_training`.

For RLVR / reasoning workloads you can skip steps 1–2 and go straight to step 3 with a custom
Python reward function (see the **First run** above and :ref:`single_turn_mode`).

Advanced modes (custom rewards, multi-turn environments, async, VLM) are all in
:doc:`agent_training`. More runnable scripts live under `examples/scripts
<https://github.com/OpenRLHF/OpenRLHF/tree/main/examples/scripts>`_.

Where to next
-------------

- **Mental model** — :doc:`architecture`, :doc:`agent_paradigm`.
- **Tuning / scaling** — :doc:`hybrid_engine`, :doc:`async_training`, :doc:`performance`,
  :doc:`multi-node`.
- **Errors** — :doc:`troubleshooting`.

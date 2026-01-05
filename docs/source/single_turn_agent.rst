Single-Turn Agent: Custom Rewards & Remote RM
==============================================

The **single-turn agent execution** (default mode) is used for 99% of RLHF use cases. It provides one-shot generation per prompt and works seamlessly with all RL algorithms.

This mode supports:

- Remote reward models via HTTP
- Custom reward functions in Python
- Standard RLHF with trained reward models

See :doc:`agent_paradigm` for architecture overview.

Remote Reward Model Server
---------------------------

Suppose we have deployed a large reward model (Llama3-405B, ArmoRM-Llama3-8B-v0.1, or PairRM) on a remote server. OpenRLHF provides an HTTP interface to use these models in RLHF training.

Starting the Remote Reward Model Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, start a remote reward model server. You can modify the example below:

`openrlhf.cli.serve_rm <https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/serve_rm.py#L1>`_

.. code-block:: python

    import argparse
    import re

    import torch
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from openrlhf.models import get_llm_for_sequence_regression
    from openrlhf.utils import get_tokenizer
    from openrlhf.utils.logging_utils import init_logger

    logger = init_logger(__name__)

    class RewardModelProxy:
        def __init__(self, args):
            # Modify the reward_model to your remote model
            self.reward_model = get_llm_for_sequence_regression(
                args.reward_pretrain,
                "reward",
                normalize_reward=args.normalize_reward,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                value_head_prefix=args.value_head_prefix,
                device_map="auto",
            )
            self.reward_model.eval()

            self.tokenizer = get_tokenizer(
                args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
            )
            self.max_length = args.max_len
            self.batch_size = args.batch_size

        def get_reward(self, queries):
            if self.batch_size is None:
                batch_size = len(queries)
            else:
                batch_size = self.batch_size

            logger.info(f"queries[0]: {queries[0]}")

            scores = []
            # batch
            with torch.no_grad():
                for i in range(0, len(queries), batch_size):
                    inputs = self.tokenize_fn(
                        queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                    )
                    r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                    r = r.tolist()
                    scores.extend(r)
            return scores

        def tokenize_fn(self, texts, device):
            batch = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )
            return {k: v.to(device) for k, v in batch.items()}


    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        # Reward Model
        parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
        parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
        parser.add_argument("--value_head_prefix", type=str, default="value_head")
        parser.add_argument("--max_len", type=int, default="2048")

        parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

        # Performance
        parser.add_argument("--load_in_4bit", action="store_true", default=False)
        parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
        parser.add_argument(
            "--attn_implementation",
            type=str,
            default="flash_attention_2",
            help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
        )
        parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=None)

        args = parser.parse_args()

        # server
        reward_model = RewardModelProxy(args)
        app = FastAPI()

        @app.post("/get_reward")
        async def get_reward(request: Request):
            data = await request.json()
            queries = data.get("query")
            rewards = reward_model.get_reward(queries)
            result = {"rewards": rewards, "scores": rewards, "extra_logs": {"dummy_scores": rewards}}
            logger.info(f"Sent JSON: {result}")
            return JSONResponse(result)

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

Launch the reward model server:

.. code-block:: bash

    python -m openrlhf.cli.serve_rm \
        --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
        --port 5000 \
        --bf16 \
        --attn_implementation flash_attention_2 \
        --normalize_reward \
        --max_len 8192 \
        --batch_size 16


Using Remote Reward Model in Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then, specify ``--remote_rm_url`` during RL training:

.. code-block:: bash

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "/openrlhf"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 2 \
        --critic_num_nodes 1 \
        --critic_num_gpus_per_node 2 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 2 \
        --vllm_num_engines 2 \
        --vllm_tensor_parallel_size 2 \
        --colocate_actor_ref \
        --ref_reward_offload \
        --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
        --remote_rm_url http://localhost:5000/get_reward \
        --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
        --micro_train_batch_size 8 \
        --train_batch_size 128 \
        --micro_rollout_batch_size 16 \
        --rollout_batch_size 1024 \
        --max_samples 100000 \
        --max_epochs 1 \
        --prompt_max_len 1024 \
        --generate_max_len 1024 \
        --packing_samples \
        --zero_stage 3 \
        --bf16 \
        --actor_learning_rate 5e-7 \
        --critic_learning_rate 9e-6 \
        --init_kl_coef 0.01 \
        --prompt_data OpenRLHF/prompt-collection-v0.1 \
        --input_key context_messages \
        --apply_chat_template \
        --normalize_reward \
        --adam_offload \
        --attn_implementation flash_attention_2 \
        --gradient_checkpointing \
        --use_wandb {wandb_token}

.. note:: We can use ``--critic_pretrain`` to specify the critic model. Otherwise the critic model is initialized using the actor model specified by ``--pretrain``.


Reinforced Fine-tuning with Custom Reward Functions
----------------------------------------------------

OpenRLHF supports convenient and efficient **Reinforced Fine-tuning** using custom reward functions. This is perfect for:

- **Rule-based rewards**: Length, format, structure checking
- **Code execution rewards**: Compile code, run test suites
- **Math verification**: Check solution correctness
- **External API rewards**: Judge models, compilers, evaluators
- **Hybrid rewards**: Combine multiple signals

Implementing Custom Reward Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You only need to implement a file containing the custom ``reward_func`` function and pass its path to the ``--remote_rm_url`` parameter.

.. code-block:: python

    # reward_func.py
    import torch

    def reward_func(queries, prompts, labels):
        """
        Compute custom rewards for generated responses.
        
        Args:
            queries: List[str] - Full text (prompt + response)
            prompts: List[str] - Original prompts only
            labels: List[str] - Ground truth labels (from --label_key)
        
        Returns:
            dict with:
                - rewards: Tensor for advantage calculation
                - scores: Tensor for dynamic filtering (0-1 range)
                - extra_logs: Dict for wandb logging
        """
        batch_size = len(queries)
        
        # Example: Random rewards (replace with your logic)
        # Real examples: code execution, math verification, format checking
        reward = torch.randint(0, 2, (batch_size,)).float()
        
        return {
            "rewards": reward,           # Used in RL advantage calculation
            "scores": reward,            # Used for dynamic filtering (--dynamic_filtering)
            "extra_logs": {              # Logged to wandb
                "custom_metric": reward.mean().item(),
            },
        }

Using Custom Reward Functions in Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``--remote_rm_url`` to the path of your reward function file:

.. code-block:: bash
    
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json='{"working_dir": "/openrlhf"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 2 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 2 \
        --vllm_num_engines 2 \
        --vllm_tensor_parallel_size 2 \
        --colocate_actor_ref \
        --pretrain meta-llama/Meta-Llama-3-8B \
        --remote_rm_url /path/to/reward_func.py \
        --label_key answer \
        --prompt_data your_prompt_dataset \
        --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
        --micro_train_batch_size 8 \
        --train_batch_size 128 \
        --micro_rollout_batch_size 16 \
        --rollout_batch_size 1024 \
        --max_samples 100000 \
        --max_epochs 1 \
        --prompt_max_len 1024 \
        --generate_max_len 1024 \
        --zero_stage 3 \
        --bf16 \
        --actor_learning_rate 5e-7 \
        --init_kl_coef 0.01 \
        --packing_samples \
        --gradient_checkpointing \
        --use_wandb {wandb_token}

.. note:: The ``--label_key`` parameter specifies the JSON key in your dataset that contains ground truth labels, which are passed to ``reward_func`` as the ``labels`` argument.

.. tip:: 
   **Use Cases**:
   
   - **Code Generation**: Execute code and run test suites for pass@k evaluation
   - **Math Problems**: Verify solution correctness against ground truth
   - **Format Checking**: Check if output follows specific structure (JSON, XML, etc.)
   - **Multi-objective**: Combine multiple reward signals (quality + length + format)

Example: Code Execution Reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # code_reward.py
    import torch
    import subprocess
    import tempfile
    import os

    def reward_func(queries, prompts, labels):
        """
        Execute generated code and reward based on test pass rate.
        """
        rewards = []
        
        for query, label in zip(queries, labels):
            # Extract code from query (assume code is between ```python and ```)
            try:
                code = query.split("```python")[1].split("```")[0].strip()
                
                # Write to temp file and execute
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Run tests
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    timeout=5
                )
                
                # Reward 1.0 if success, 0.0 if failure
                reward = 1.0 if result.returncode == 0 else 0.0
                
                os.unlink(temp_file)
                
            except Exception as e:
                reward = 0.0
            
            rewards.append(reward)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        return {
            "rewards": rewards_tensor,
            "scores": rewards_tensor,
            "extra_logs": {
                "pass_rate": rewards_tensor.mean().item(),
            },
        }

.. warning::
   Custom reward functions should handle errors gracefully and return valid tensors for all inputs.

Algorithm Compatibility
------------------------

All custom reward approaches (remote RM or custom functions) work with **any RL algorithm**. Switch algorithms via ``--advantage_estimator``. See :doc:`agent_paradigm` for available algorithms and :doc:`rl` for detailed usage.


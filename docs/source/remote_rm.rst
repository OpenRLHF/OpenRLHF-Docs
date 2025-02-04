Remote Reward Model
=====

How to specify the reward model on a remote server?
------------

Suppose we have a scenario where we have deployed Llama3-405B, ArmoRM-Llama3-8B-v0.1, or a PairRM on a remote server. 
We want to call these reward model services in RLHF. OpenRLHF provides an HTTP interface to achieve this. 

First, we need to start a remote reward model server on the remote server, which can be modified using the example below.

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


    def strip_sequence(text, pad_token, eos_token):
        pad_token_escaped = re.escape(pad_token)
        eos_token_escaped = re.escape(eos_token)

        pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
        text = re.sub(pattern, "", text)

        pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
        text = re.sub(pattern, "", text)
        return text


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

            # remove pad_token
            for i in range(len(queries)):
                queries[i] = (
                    strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                    + self.tokenizer.eos_token
                )
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
        parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
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
            result = {"rewards": rewards}
            logger.info(f"Sent JSON: {result}")
            return JSONResponse(result)

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

Launch the reward model server

.. code-block:: shell

    python -m openrlhf.cli.serve_rm \
        --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
        --port 5000 \
        --bf16 \
        --flash_attn \
        --normalize_reward \
        --max_len 8192 \
        --batch_size 16


Then, we can specify ``remote_rm_urls`` during PPO training.

.. code-block:: shell

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
        --flash_attn \
        --gradient_checkpointing \
        --use_wandb {wandb_token}

.. note:: We can use ``--critic_pretrain`` to specify the critic model. Otherwise the critic model is initialized using the actor model specified by ``--pretrain``.


Reinforced Fine-tuning
------------

OpenRLHF supports convenient and efficient Reinforced Fine-tuning. You only need to implement a `file containing the custom reward_fun function <https://github.com/OpenRLHF/OpenRLHF/blob/custom_reward_func/examples/scripts/reward_func.py>`_ and pass its path to the ``remote_rm_url`` parameter. Such as

.. code-block:: python

    # reward_func.py
    import torch

    def reward_func(queries, prompts):
        # queries is prompts + responses
        print(queries)
        return torch.randn(len(queries))


then just set

.. code-block:: shell
    
    ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    ...
    --remote_rm_url /path/to/reward_func.py

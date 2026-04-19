Multi-node Training
===================

How to launch Ray PPO on Slurm?
-----------------------------------

Here is an example (uses the 0.10.2 hierarchical CLI):

.. code-block:: bash

    #!/bin/bash

    #SBATCH -p { partition }
    #SBATCH -A { account }
    #SBATCH -J { jobname }
    #SBATCH -N 2                       # 64x8x4
    #SBATCH -t {LIMIT_TIME}            # wall time
    #SBATCH --ntasks-per-node=1        # tasks per node
    #SBATCH --exclusive                # exclusive node access
    #SBATCH --mem=0                    # all mem avail
    #SBATCH --mail-type=FAIL           # only send email on failure
    #SBATCH --overcommit               # needed for pytorch

    # project settings
    OPENRLHF_PATH=<OPENRLHF_ROOT_PATH>
    MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
    IMAGE_NAME="nvcr.io/nvidia/pytorch:25.11-py3"
    RAY_VERSION=2.12.0

    JOBLOG="$(realpath .)/train_ppo_llama_ray-$SLURM_JOB_ID.log"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

    # launch ray daemon
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
    nodes_array=( $nodes )
    node_1=${nodes_array[0]}
    ip=$node_1

    port=6379
    ip_head=$ip:$port
    export ip_head
    echo "IP Head: $ip_head"  &>> ${JOBLOG}

    echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
    sleep 10s

    worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
    for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
        "pip install ray[default]==$RAY_VERSION \
        && /root/.local/bin/ray start --address $ip_head --block" &>> ${JOBLOG} &
    sleep 1s;
    done

    sleep 30s

    # ===== submit ray job =====
    # Job start
    srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray job submit --address=http://localhost:8265 \
        --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref.num_nodes 1 \
        --ref.num_gpus_per_node 4 \
        --reward.num_nodes 1 \
        --reward.num_gpus_per_node 4 \
        --critic.num_nodes 1 \
        --critic.num_gpus_per_node 4 \
        --actor.num_nodes 1 \
        --actor.num_gpus_per_node 4 \
        --vllm.num_engines 4 \
        --vllm.tensor_parallel_size 2 \
        --train.colocate_critic_reward \
        --train.colocate_actor_ref \
        --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
        --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-mixture \
        --ckpt.output_dir /openrlhf/examples/checkpoint/llama3-8b-rlhf \
        --train.micro_batch_size 8 \
        --train.batch_size 128 \
        --rollout.micro_batch_size 16 \
        --rollout.batch_size 1024 \
        --data.max_samples 100000 \
        --train.max_epochs 1 \
        --data.max_len 2048 \
        --ds.zero_stage 3 \
        --ds.param_dtype bf16 \
        --actor.adam.lr 5e-7 \
        --critic.adam.lr 9e-6 \
        --algo.kl.init_coef 0.01 \
        --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
        --data.input_key context_messages \
        --data.apply_chat_template \
        --reward.normalize_enable \
        --vllm.sync_backend nccl \
        --ds.packing_samples \
        --ds.adam_offload \
        --actor.gradient_checkpointing_enable \
        --logger.wandb.key {wandb_token}" &>> ${JOBLOG}

    echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}


How to launch SFT/RM/DPO training on Slurm?
----------------------------------------------

Here is an example for DPO:

.. code-block:: bash

    #!/bin/bash

    #SBATCH -p { partition }
    #SBATCH -A { account }
    #SBATCH -J { jobname }
    #SBATCH -N 1                      # 64x8x4
    #SBATCH -t 0-4:00:00             # wall time
    #SBATCH --ntasks-per-node=1       # tasks per node
    #SBATCH --exclusive                # exclusive node access
    #SBATCH --mem=0                    # all mem avail
    #SBATCH --mail-type=FAIL           # only send email on failure
    #SBATCH --overcommit               # needed for pytorch

    OPENRLHF_PATH=<OPENRLHF_ROOT_PATH>
    IMAGE_NAME="nvcr.io/nvidia/pytorch:25.11-py3"
    MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
    GPUS_PER_NODE=8
    JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"

    readonly training_commands=" \
        openrlhf.cli.train_dpo \
        --ckpt.output_dir ./checkpoint/llama3-8b-dpo \
        --ckpt.save_steps -1 \
        --logger.logging_steps 1 \
        --eval.steps -1 \
        --train.batch_size 256 \
        --train.micro_batch_size 1 \
        --model.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
        --ds.param_dtype bf16 \
        --train.max_epochs 1 \
        --data.max_len 8192 \
        --ds.zero_stage 3 \
        --adam.lr 5e-7 \
        --model.beta 0.1 \
        --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
        --data.apply_chat_template \
        --data.chosen_key chosen \
        --data.rejected_key rejected \
        --ds.packing_samples \
        --ds.attn_implementation flash_attention_2 \
        --model.gradient_checkpointing_enable \
        --logger.wandb.key {wandb_token}"

    echo $training_commands &>> ${JOBLOG}

    # Job start
    echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

    # master addr and port
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=9901

    srun --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
        "cd /openrlhf; pip install . ; torchrun \
        --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
        --master_addr $MASTER_ADDR --master_port $MASTER_PORT -m ${training_commands}" &>> ${JOBLOG}

    echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}


How to specify a particular node for a model in Ray PPO?
------------------------------------------------------------

In Ray, you can control task scheduling by specifying the node's IP address. Ray allows you to specify resource constraints when submitting tasks, and you can use custom resource labels to help the scheduler select the appropriate node. Here is a basic example demonstrating how to use Ray's resource labels and IP addresses to specify nodes:

Start Ray on the nodes:

When starting Ray on each node, you can specify custom resource labels. For example:

On machine A (with V100):

.. code-block:: bash

    ray start --node-ip-address=<IP of machine A> --resources '{"v100": 1}'

On machine B (with H100):

.. code-block:: bash

    ray start --node-ip-address=<IP of machine B> --resources '{"h100": 1}'

Specify resource requirements in your script:

When submitting tasks, you can specify the resources required for the task. For example:

.. code-block:: python

    import ray
    from ray.util.placement_group import placement_group

    ray.init(address='auto')

    # Create Placement Groups
    pg = placement_group([{"v100": 1, "h100": 1}])
    ray.get([pg.ready()])

    @ray.remote
    def reference_or_reward_model():
        # Task suitable for small GPU memory
        pass

    @ray.remote
    def actor_or_critic_model():
        # Task suitable for large GPU memory
        pass

    # Launch Tasks
    result1 = reference_or_reward_model.options(placement_group=pg, resources={"v100": 1}).remote()
    result2 = actor_or_critic_model.options(placement_group=pg, resources={"h100": 1}).remote()

In this example, task1 will be scheduled on a node with the small GPU memory resource (i.e., machine A), and task2 will be scheduled on a node with the large GPU memory resource (i.e., machine B).

Based on this, you can modify the ``Ray resources``-related code in `train_ppo_ray.py <https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/train_ppo_ray.py>`_.
For example, we want to deploy the ``Reference Model`` on a ``V100 x8`` node (other models on ``A100 x8``):

.. code-block:: python

    ray start --node-ip-address=<IP of machine A> --resources '{"v100": 8}'

    # Modify
    ref_model = PPORayActorGroup(
            args.ref.num_nodes,
            args.ref.num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.25 if pg else 1,
        )

    # To
    # Do not use --train.colocate_actor_ref for the models
    ref_model = PPORayActorGroup(
            args.ref.num_nodes,
            args.ref.num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=1,
            resources={"v100": 1}
            num_resources_per_node=8,
        )

.. note:: `Ray resources docs <https://docs.ray.io/en/latest/ray-core/scheduling/resources.html>`_

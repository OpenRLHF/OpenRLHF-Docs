FQA
=====

How to specifies the nodes allocation for the models in Ray PPO.
------------

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
    v100_pg = placement_group([{"v100": 1}])
    h100_pg = placement_group([{"h100": 1}])
    ray.get([v100_pg.ready(), h100_pg.ready()])

    @ray.remote(resources={"v100": 1})
    def reference_or_reward_model():
        # Task suitable for small GPU memory
        pass

    @ray.remote(resources={"h100": 1})
    def actor_or_critic_model():
        # Task suitable for large GPU memory
        pass

    # Launch Tasks
    result1 = reference_or_reward_model.options(placement_group=v100_pg).remote()
    result2 = actor_or_critic_model.options(placement_group=h100_pg).remote()
    
In this example, task1 will be scheduled on a node with the small GPU memory resource (i.e., machine A), and task2 will be scheduled on a node with the large GPU memory resource (i.e., machine B).
Based on this, you can modify the node ``placement_group`` and ``resources``-related code in `train_ppo_ray.py <https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/train_ppo_ray.py>`_.


How to launch Ray PPO on Slurm.
------------

Here is an example

.. code-block:: bash

    #!/bin/bash

    #SBATCH -A {ACCOUNT}
    #SBATCH -J openrlhf-ppo-ray
    #SBATCH -N 2                       # 64x8x4
    #SBATCH -t {LIMIT_TIME}            # wall time
    #SBATCH --ntasks-per-node=1        # tasks per node
    #SBATCH --exclusive                # exclusive node access
    #SBATCH --mem=0                    # all mem avail
    #SBATCH --mail-type=FAIL           # only send email on failure
    #SBATCH --overcommit               # needed for pytorch

    # project settings
    PROJECT_PATH=<OPENRLHF_ROOT_PATH>
    MOUNT="$PROJECT_PATH:/openrlhf,$HOME/.cache:/root/.cache"
    IMAGE_NAME="nvcr.io/nvidia/pytorch:24.02-py3"
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
    && pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block" &>> ${JOBLOG} &
    sleep 10s

    worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
    for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
        && pip install ray[default]==$RAY_VERSION \
        && /root/.local/bin/ray start --address "$ip_head" --block" &>> ${JOBLOG} &
    sleep 1s;
    done

    sleep 30s

    # ===== submit ray job =====
    # Job start
    srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "pip install ray[default]==$RAY_VERSION \
    && /root/.local/bin/ray job submit --address=http://localhost:8265 \
        --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
        -- python3 examples/train_ppo_ray.py \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 2 \
        --reward_num_nodes 1 \
        --reward_num_gpus_per_node 2 \
        --critic_num_nodes 1 \
        --critic_num_gpus_per_node 2 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 2 \
        --vllm_num_engines 2 \
        --vllm_tensor_parallel_size 2 \
        --colocate_critic_reward \
        --colocate_actor_ref \
        --ref_reward_offload \
        --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
        --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
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
        --critic_learning_rate 9e-6 \
        --init_kl_coef 0.01 \
        --prompt_data OpenLLMAI/prompt-collection-v0.1 \
        --input_key context_messages \
        --apply_chat_template \
        --normalize_reward \
        --adam_offload \
        --flash_attn \
        --gradient_checkpointing \
        --use_wandb {wandb_token}" &>> ${JOBLOG}

    echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
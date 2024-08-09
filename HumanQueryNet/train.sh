config=human_query_net_all_task_r50_scratch
python -m torch.distributed.run \
            --nnodes=$WORLD_SIZE   \
            --node_rank=$RANK  \
            --master_addr=$MASTER_ADDR  \
            --nproc_per_node=8 \
            --master_port=$MASTER_PORT   \
            ./tools/train.py ./configs/${config}.py --gpus=16 --launcher pytorch  --work-dir=workdir/${config}
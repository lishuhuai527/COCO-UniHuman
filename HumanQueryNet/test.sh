job_name=human_query_net_all_task_r50_scratch
config=${job_name}.py
ckpt=HumanQueryNet_r50.pth
python -m torch.distributed.run \
            --nnodes=1   \
            --node_rank=0  \
            --master_addr=127.0.0.1  \
            --nproc_per_node=8 \
            --master_port=8088   \
            ./tools/test.py ./configs/{} workdir/${job_name}/${ckpt}  --launcher=pytorch --work-dir=workdir/${job_name}/ --eval all
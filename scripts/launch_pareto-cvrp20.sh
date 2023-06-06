#!/bin/bash

RUN_CMD="python scripts/eval_methods_pareto.py"

# CVRP20
SIZE=20
TASK="cvrp${SIZE}"
GPU_ID=2
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/pomo-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/symnco-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-${TASK}-sm-xl" --gpu_id ${GPU_ID} --checkpoint "epoch_499.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-${TASK}-sm" --gpu_id ${GPU_ID} --checkpoint "epoch_499.ckpt"

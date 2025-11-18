#!/bin/bash

export DATA_DIR
export CKPT_DIR
export NSIGHT_LOG_DIR ="${NSIGHT_LOG_DIR:-nsight_logs}"
export NSIGHT_FILE_NAME
export MODE
export LOCAL_GPU_IDS
export NUM_GPUS

###########################################################################
# Problem 0: Generate Nsight log
# Find the correct way to generate Nsight log.
###########################################################################

# (1) Create log DIR if not exist
mkdir -p "${NSIGHT_LOG_DIR}"


# (2) Run training with Nsight Systems profiling enabled
CUDA_VISIBLE_DEVICES="${LOCAL_GPU_IDS}" nsys profile \
    --output "${NSIGHT_LOG_DIR%/}/${NSIGHT_FILE_NAME}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,cublas,cudnn,osrt \
  python train_cifar.py \
    --num_gpu=$NUM_GPUS \
    --data="$DATA_DIR" \
    --ckpt="$CKPT_DIR" \
    --mode="$MODE" \
    --save_ckpt

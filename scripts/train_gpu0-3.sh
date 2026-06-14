#!/usr/bin/env bash
# DGNO distributed training on GPUs 0-3.
# Usage: bash scripts/train_gpu0-3.sh options/DGNO_BBBC006_w1_Face.yml
set -e
cd "$(dirname "$0")/.."

CONFIG=$1
if [ -z "$CONFIG" ]; then
  echo "Usage: bash scripts/train_gpu0-3.sh <config.yml>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DISTRIBUTED_DEBUG=DETAIL \
  python -m torch.distributed.run --nproc_per_node=4 --master_port=9960 \
  basicsr/train.py -opt "$CONFIG" --launcher pytorch

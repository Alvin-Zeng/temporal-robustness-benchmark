#!/bin/bash
echo "start training"
# export CUDA_VISABLE_DEVICES=3
python train.py ./configs/thumos_i3d.yaml --output random1_jq
#!/bin/bash
echo "start training"
# export CUDA_VISABLE_DEVICES=0
python train.py ./configs/thumos_videomae.yaml --output random1_overexposure_w_1

#!/bin/bash

# 设置预训练模型路径
PRETRAINED_CKPT="/data/housen/Unet/medium/train/UNet128_cifar100-epoch_550-timesteps_1000-class_condn_-1_550.pt"

for i in {0..99}
do
    CUDA_VISIBLE_DEVICES=3,4,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port 8110 main.py \
        --arch UNet --dataset cifar100 --class-cond $i --sampling-only --sampling-steps 250 \
        --num-sampled-images 500 --pretrained-ckpt $PRETRAINED_CKPT --ddim --batch-size 500 --save-dir "/data/housen/Unet/medium/800"
done
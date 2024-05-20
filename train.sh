# CUDA_VISIBLE_DEVICES=0,4,6,7 python -m torch.distributed.launch --nproc_per_node=4 main.py \
#   --arch UNet --dataset cifar100 --class-cond -1 --epochs 500 --batch-size 512 --ddim

# CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 main.py \
#   --arch UNetSmall --dataset cifar100 --class-cond -1 --epochs 500 --batch-size 1920 --ddim

CUDA_VISIBLE_DEVICES=1,3,4 python -m torch.distributed.launch --nproc_per_node=3 --master_port 8102 main.py \
  --arch UNet --dataset cifar100 --class-cond -1 --epochs 300 --batch-size 1024 --ddim --save-dir /data/housen/Unet/medium/train --pretrained-ckpt trained_models/UNet_cifar100-epoch_500-timesteps_1000-class_condn_-1.pt
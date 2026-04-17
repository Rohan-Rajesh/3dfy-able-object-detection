#!/bin/bash
#SBATCH --job-name=videomt_custom
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=7:00:00
#SBATCH --output=output_videomt_custom_ViTL/slurm_%j.out
#SBATCH --error=output_videomt_custom_ViTL/slurm_%j.err

export WANDB_API_KEY="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"

PYTHON=/gscratch/krishna/rohanr12/miniconda3/envs/videomt/bin/python

cd /mmfs1/gscratch/krishna/rohanr12/videomt

$PYTHON train_net_video.py \
    --num-gpus 1 \
    --config-file configs/custom/videomt_custom_ViTL.yaml \
    MODEL.WEIGHTS weights/yt_2019_vit_large.pth

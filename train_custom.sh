#!/bin/bash
#SBATCH --job-name=videomt
#SBATCH --account=krishna
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=output_videomt_custom_ViTL/slurm_%j.out
#SBATCH --error=output_videomt_custom_ViTL/slurm_%j.err

export WANDB_API_KEY="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"

PYTHON=/gscratch/krishna/rohanr12/miniconda3/envs/videomt/bin/python

cd /mmfs1/gscratch/krishna/rohanr12/videomt

$PYTHON train_net_video.py \
      --num-gpus 4 \
      --config-file configs/custom/videomt_custom_ViTL.yaml \
      MODEL.WEIGHTS weights/yt_2022_vit_large.pth

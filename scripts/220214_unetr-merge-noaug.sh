#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH -J BRATS-UNETR-MERGE-4LAYER-NOAUG
#SBATCH -p all
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.e

# Environment variables

# Logging
export BATCH_SIZE_TRAIN=64
export BATCH_SIZE_TEST=64
export NUM_EPOCHS=50
export VAL_FREQ=2
export ROOT_DIR="/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
export CKPT_SAVE_DIR="./result/exps/unetr-merge-4layer-noaug"

# Model configuration
export IN_CHANNELS=4
export OUT_CHANNELS=4
export IMG_SIZE=240
export FEATURE_SIZE=8
export HIDDEN_SIZE=64
export NUM_HEADS=4
export DROPOUT_RATE=0.3
export MLP_DIM=256
export NORM_NAME="instance"
export SPATIAL_DIMS=2

# Loss function
export LAMBDA_CE=0.7
export LAMBDA_DICE=0.3

# Optimizer
export LR=0.01

python train_unetr.py

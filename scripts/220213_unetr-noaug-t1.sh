#!/bin/bash

#SBATCH -t 72:00:00
#SBATCH --mem=64G
#SBATCH -J BRATS-UNETR-T1-NOAUG
#SBATCH -p all
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.e

# Environment variables

# Logging
export BATCH_SIZE_TRAIN=128
export BATCH_SIZE_TEST=64
export NUM_EPOCHS=10
export VAL_FREQ=2
export ROOT_DIR="/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"
export CKPT_SAVE_DIR="./result/exps/unetr-t1-noaug-tmp"

# Model configuration
export IN_CHANNELS=1
export OUT_CHANNELS=4
export IMG_SIZE=240
export FEATURE_SIZE=8
export HIDDEN_SIZE=64
export NUM_HEADS=4
export DROPOUT_RATE=0.3
export MLP_DIM=512
export NORM_NAME="batch"
export SPATIAL_DIMS=2

# Loss function
export LAMBDA_CE=0.8
export LAMBDA_DICE=0.2

# Optimizer
export LR=0.01

python train_unetr.py

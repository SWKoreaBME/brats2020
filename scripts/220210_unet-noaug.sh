#!/bin/bash

#SBATCH -t 72:00:00
#SBATCH --mem=64G
#SBATCH -J BRATS-UNET-NOAUG
#SBATCH -p all
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.e

python train_unet.py

#!/bin/bash

#SBATCH -t 72:00:00
#SBATCH --mem=64G
#SBATCH -J BRATS-UNETR-NOAUG
#SBATCH -p all
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.e

python train_unetr.py

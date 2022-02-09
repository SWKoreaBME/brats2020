#!/bin/bash

#SBATCH -t 0-24:00:00
#SBATCH --mem=16G
#SBATCH -J Transform-BraTs2020
#SBATCH -p all
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -o ./out/%x-%j.out
#SBATCH -e ./out/%x-%j.e

python transform.py
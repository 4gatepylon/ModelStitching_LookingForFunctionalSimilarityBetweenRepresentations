#!/bin/bash

#SBATCH --gres=gpu:volta:1
#SBATCH -o myScript.sh.log-%j-%a
#SBATCH -a 1-256

python3 -O cifar_supervised.py --train_mode stitchtrain_small --smallpairnum $SLURM_ARRAY_TASK_ID

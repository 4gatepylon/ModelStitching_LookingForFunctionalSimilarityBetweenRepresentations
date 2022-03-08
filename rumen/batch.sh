#!/bin/bash

#SBATCH -o myScript.sh.log-%j-%a
#SBATCH -a 1-256

python3 -O cifar_supervised.py --smallpairnum $SLURM_ARRAY_TASK_ID

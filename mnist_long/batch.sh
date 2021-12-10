#!/bin/bash
module load anaconda/2021a && module load cuda/11.2 && LLsub batch0.sh -s 20 -g volta:1 && LLsub batch1.sh -s 20 -g volta:1
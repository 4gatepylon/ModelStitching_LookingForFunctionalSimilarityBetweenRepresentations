#!/bin/bash
module load anaconda/2021a && module load cuda/11.2 && LLsub sanity_batch.sh -s 20 -g volta:1
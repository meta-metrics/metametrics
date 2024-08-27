#!/bin/bash -l
#$ -P multilm
#$ -l gpus=1
#$ -l gpu_type=A100
#$ -l gpu_memory=80G
#$ -N xxl_metametrics

module load python3/3.10
python3 -m tasks.wmt24.generate_score_xxl

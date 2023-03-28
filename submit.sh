#!/bin/bash
#SBATCH --job-name=aggregate-perf-0-4
#SBATCH --output=per.txt
#SBATCH --time=365-00:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s5
#SBATCH --cpus-per-task=16
#SBATCH --gpus=8
#SBATCH --mem=100GB

 python train.py -m "google/flan-t5-large" --train --overwrite-cache -b 2 --accum 4

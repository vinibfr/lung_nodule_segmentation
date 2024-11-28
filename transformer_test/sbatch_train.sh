#!/bin/bash
#
#SBATCH --partition=gpu_min12gb
#SBATCH --qos=gpu_min12gb
#SBATCH --job-name=lungISTR
#SBATCH -o slurm.full.%N.%j.out
#SBATCH -e slurm.full.%N.%j.err

python3 /nas-ctm01/homes/vbreis/codes/transformer_test/swin_train.py
exit 0

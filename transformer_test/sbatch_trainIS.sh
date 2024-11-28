#!/bin/bash
#
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=lungISTR
#SBATCH -o slurm.IS.%N.%j.out
#SBATCH -e slurm.IS.%N.%j.err

python3 /nas-ctm01/homes/vbreis/codes/transformer_test/swin_trainIS.py
exit 0

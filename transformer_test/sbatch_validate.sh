#!/bin/bash
#
#SBATCH --partition=gpu_min80GB
#SBATCH --qos=gpu_min80GB
#SBATCH --job-name=lungseg
#SBATCH -o slurm.validate.Hyper.%N.%j.out
#SBATCH -e slurm.validate.Hyper.%N.%j.err

python3 /nas-ctm01/homes/vbreis/codes/transformer_test/validate.py
exit 0

#!/bin/bash
#SBATCH --job-name=resid_sweep
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=resid_sweep_out_%j.txt
#SBATCH --error=resid_sweep_err_%j.txt

export PYTHONUNBUFFERED=1
cd /lustre/fs1/home/br123310/Electromagnetic_Solvers-main
/home/br123310/.conda/envs/pinns/bin/python residual_sweep.py
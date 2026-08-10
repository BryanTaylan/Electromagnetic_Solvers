#!/bin/bash
#SBATCH --job-name=bc_compare
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=bc_compare_out_%j.txt
#SBATCH --error=bc_compare_err_%j.txt

export PYTHONUNBUFFERED=1
cd /lustre/fs1/home/br123310/Electromagnetic_Solvers-main
/home/br123310/.conda/envs/pinns/bin/python bc_comparison.py
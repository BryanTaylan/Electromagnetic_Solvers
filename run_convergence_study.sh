#!/bin/bash
#SBATCH --job-name=converge
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=converge_out_%j.txt
#SBATCH --error=converge_err_%j.txt

export PYTHONUNBUFFERED=1
cd /lustre/fs1/home/br123310/Electromagnetic_Solvers-main
/home/br123310/.conda/envs/pinns/bin/python convergence_study.py
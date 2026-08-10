#!/bin/bash
#SBATCH --job-name=sep_conv
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=sep_conv_out_%j.txt
#SBATCH --error=sep_conv_err_%j.txt

export PYTHONUNBUFFERED=1
cd /lustre/fs1/home/br123310/Electromagnetic_Solvers-main

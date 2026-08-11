#!/bin/bash
#SBATCH --job-name=stability_probe
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=stability_probe_out_%j.txt
#SBATCH --error=stability_probe_err_%j.txt

export PYTHONUNBUFFERED=1
cd /lustre/fs1/home/br123310/Electromagnetic_Solvers-main
/home/br123310/.conda/envs/pinns/bin/python stability_probe.py
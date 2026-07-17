#!/bin/bash
#SBATCH --job-name=scene_abl
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/scene_abl_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
date; python -u scripts/scene_ablation.py || { echo "SCENE_ABL FAILED"; exit 1; }; date

#!/bin/bash
#SBATCH --job-name=calib2d
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/calib2d_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/probblend_calib2d.py 2>&1 | grep -v "module command"
echo CALIB2D_JOB_DONE

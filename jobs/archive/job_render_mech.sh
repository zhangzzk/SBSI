#!/bin/bash
#SBATCH --job-name=rmech
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rmech_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date; stdbuf -oL -eL python -u scripts/render_diff_mechanism.py 2>&1 | grep -v "module command"; date; echo RMECH_JOBDONE

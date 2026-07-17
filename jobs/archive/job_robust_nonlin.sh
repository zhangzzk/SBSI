#!/bin/bash
#SBATCH --job-name=robnl
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/robnl_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date; stdbuf -oL -eL python -u scripts/robust_nonlin.py 2>&1 | grep -v "module command"; date; echo ROBNL_JOBDONE

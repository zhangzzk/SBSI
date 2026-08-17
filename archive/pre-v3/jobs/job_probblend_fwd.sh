#!/bin/bash
#SBATCH --job-name=pbfwd
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/pbfwd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
date
python -u scripts/probblend_forward.py --cases 0 1 2 3 --n-prim 30000 --n-syn 40 \
  2>&1 | grep -v "module command"
date
echo PBFWD_JOB_DONE

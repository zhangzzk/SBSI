#!/bin/bash
#SBATCH --job-name=ctxdiag
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ctxdiag_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/probblend_ctx_diag.py 2>&1 | grep -v "module command"
echo CTXDIAG_JOB_DONE

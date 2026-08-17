#!/bin/bash
#SBATCH --job-name=gapdiag
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/gapdiag_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/probblend_gap_diag.py 2>&1 | grep -v "module command"
echo GAPDIAG_JOB_DONE

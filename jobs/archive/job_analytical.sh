#!/bin/bash
#SBATCH --job-name=analytic
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/analytic_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
date; python -u scripts/analytical_combiner.py || { echo ANALYTIC_FAILED; exit 1; }; date

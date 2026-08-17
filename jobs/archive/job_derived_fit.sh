#!/bin/bash
#SBATCH --job-name=derivfit
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/derivfit_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; python -u scripts/derived_fit.py || { echo ANALYTIC_FAILED; exit 1; }; date

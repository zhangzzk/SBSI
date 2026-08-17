#!/bin/bash
#SBATCH --job-name=toy_cp
#SBATCH --time=00:40:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/toy_closepair_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; python -u scripts/toy_close_pair_scan.py --nreal 400 || { echo TOY_CP_FAILED; exit 1; }; date
echo TOY_CP_JOB_DONE

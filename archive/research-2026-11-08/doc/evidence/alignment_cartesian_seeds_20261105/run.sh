#!/bin/bash
#SBATCH --partition=inter,cip
#SBATCH --job-name=alignment_seed_repeat
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_cartesian_seeds_20261105/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(612 613)
arm=${1:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_cartesian_seeds_20261105/train.py "$arm"

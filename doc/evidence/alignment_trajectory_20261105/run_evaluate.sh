#!/bin/bash
#SBATCH --partition=cip
#SBATCH --job-name=trajectory_evaluate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_trajectory_20261105/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(fs2 galsbi galsbi_fs2rate)
arm=${1:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_trajectory_20261105/evaluate.py "$arm"

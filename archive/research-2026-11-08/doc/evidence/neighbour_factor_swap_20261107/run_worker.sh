#!/bin/bash
#SBATCH --partition=cip,inter
#SBATCH --job-name=factor_swap_worker
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_factor_swap_20261107/logs/worker_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_factor_swap_20261107/pipeline.py worker

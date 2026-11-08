#!/bin/bash
#SBATCH --job-name=photo_fixed_cell
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_fixed_cell_20261106/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/photometry_fixed_cell_20261106/evaluate.py

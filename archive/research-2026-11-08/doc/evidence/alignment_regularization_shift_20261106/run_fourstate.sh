#!/bin/bash
#SBATCH --partition=cip,inter
#SBATCH --exclude=met-cl-vis02
#SBATCH --job-name=guard_shift_mixed
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularization_shift_20261106/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_regularization_shift_20261106/evaluate_fourstate.py

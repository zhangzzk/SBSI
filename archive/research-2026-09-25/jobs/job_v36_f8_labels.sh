#!/bin/bash
#SBATCH --job-name=sbsi-f8-labels
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v36_f8_labels_%j.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /tmp
"$ROOT/venv/bin/python" -u "$ROOT/run_v36_f8_stage.py" --root "$ROOT" --stage labels

#!/bin/bash
#SBATCH --job-name=sbsi-c12-check
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v36_c12_check_%j.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925
"$ROOT/venv/bin/python" -m pytest -q tests/test_v36_classifier_ablation.py
"$ROOT/venv/bin/python" -u /home/z/Zekang.Zhang/SBSI/scripts/check_v36_classifier_ablation.py

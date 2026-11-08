#!/bin/bash
#SBATCH --job-name=sbsi-c12-summary
#SBATCH --partition=cip
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/v36_c12_summary_%j.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925
"$ROOT/venv/bin/python" -u -m scripts.summarize_v36_classifier_ablation --root "$ROOT"

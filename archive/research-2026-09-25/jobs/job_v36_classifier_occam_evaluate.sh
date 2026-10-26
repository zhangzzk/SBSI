#!/bin/bash
#SBATCH --job-name=sbsi-c12-eval
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v36_c12_eval_%A_%a.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925
"$ROOT/venv/bin/python" -u -m scripts.run_v36_classifier_ablation --root "$ROOT" --stage evaluate --index "$SLURM_ARRAY_TASK_ID"

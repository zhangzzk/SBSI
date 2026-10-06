#!/bin/bash
#SBATCH --job-name=sbsi-c12-eval-fix
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v36_c12_eval_fixed_%A_%a.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CLASSIFIER_SCORING_RUNNER="$ROOT/run_v36_classifier_evaluation_fixed.py"
unset PYTHONPATH
cd /tmp
"$ROOT/venv/bin/python" -m pytest -q /home/z/Zekang.Zhang/SBSI/tests/test_v36_classifier_scoring.py
"$ROOT/venv/bin/python" -u "$SBSI_CLASSIFIER_SCORING_RUNNER" --root "$ROOT" --stage evaluate --index "$SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=sbsi-c12-train
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v36_c12_train_%A_%a.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_occam_20260925_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_occam_20260925
PY="$ROOT/venv/bin/python"
"$PY" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$ROOT/resources_${SLURM_JOB_ID}.json"
"$PY" -m pytest -q tests/test_v36_classifier_ablation.py
"$PY" -u -m scripts.run_v36_classifier_ablation --root "$ROOT" --stage train --index "$SLURM_ARRAY_TASK_ID"

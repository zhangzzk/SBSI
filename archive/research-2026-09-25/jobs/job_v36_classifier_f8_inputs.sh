#!/bin/bash
#SBATCH --job-name=sbsi-c8
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v36_c8_%j.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_f8_inputs_20260925_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_f8_inputs_20260925
PY="$ROOT/venv/bin/python"
case "$1" in
  prepare)
    "$PY" -m pytest -q tests/test_v36_classifier_f8_inputs.py tests/test_v36_classifier_acceptance_amendment.py
    "$PY" -u -m scripts.prepare_v36_classifier_f8_inputs --root "$ROOT"
    ;;
  train)
    "$PY" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$ROOT/resources_${SLURM_JOB_ID}.json"
    "$PY" -u -m scripts.run_v36_classifier_ablation --root "$ROOT" --stage train --index "$SLURM_ARRAY_TASK_ID"
    ;;
  evaluate)
    "$PY" -u -m scripts.run_v36_classifier_ablation --root "$ROOT" --stage evaluate --index "$SLURM_ARRAY_TASK_ID"
    ;;
  summary)
    "$PY" -u -m scripts.summarize_v36_classifier_ablation --root "$ROOT"
    ;;
  *) exit 2 ;;
esac

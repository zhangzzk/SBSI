#!/bin/bash
#SBATCH --job-name=sbsi-f8
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v36_f8_%j.out
set -euo pipefail
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
PY="$ROOT/venv/bin/python"
cd /tmp
"$PY" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$ROOT/resources_${SLURM_JOB_ID}.json"
ARGS=(--root "$ROOT" --stage "$1")
if [[ "$1" == evaluate ]]; then
  ARGS+=(--chunk "$SLURM_ARRAY_TASK_ID")
fi
"$PY" -u "$ROOT/run_v36_f8_stage.py" "${ARGS[@]}"

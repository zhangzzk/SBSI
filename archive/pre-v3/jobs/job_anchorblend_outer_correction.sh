#!/bin/bash
#SBATCH --job-name=ab_outerfit
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_outerfit_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
OUT=results/anchorblend_outer_correction_c200-299.json
MODEL=results/anchorblend_outer_correction_c200-299.joblib
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
[ ! -e "$MODEL" ] || { echo "REFUSING existing $MODEL"; exit 1; }
"$PY" -u scripts/fit_anchorblend_outer_correction.py \
  --all-response results/anchorblend_g005_response_v22_c100-299.feather \
  --local-response results/anchorblend_g005_local10_response_v22_c200-299.feather \
  --outer-features results/anchorblend_outer_features_c200-299.feather \
  --development-max 249 --seed 25876 --output-json "$OUT" --output-model "$MODEL"

echo ANCHORBLEND_OUTER_CORRECTION_JOB_DONE; date

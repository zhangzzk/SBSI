#!/bin/bash
#SBATCH --job-name=ab_o3an
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_o3an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_o3an_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
OUT=results/anchorblend_other3abs_v22_c200-299.json
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_other3abs.py \
  --coherent results/anchorblend_g005_other3abs_c200-299.feather \
  --random results/anchorblend_random_local10_other3abs_c200-299.feather \
  --random results/anchorblend_random_layer1_local10_other3abs_c200-299.feather \
  --random results/anchorblend_random_layer2_local10_other3abs_c200-299.feather \
  --development-max 249 --output "$OUT"
echo ANCHORBLEND_OTHER3ABS_ANALYSIS_JOB_DONE
date

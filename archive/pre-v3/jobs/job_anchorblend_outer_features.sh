#!/bin/bash
#SBATCH --job-name=ab_outerx
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_outerx_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext
OUT=results/anchorblend_outer_features_c200-299.feather
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/compute_anchorblend_outer_features.py \
  --anchors results/anchorblend_g005_local10_response_v22_c200-299.feather \
  --base "$BASE" --case-min 200 --case-max 299 --n-jobs 12 --output "$OUT"

echo ANCHORBLEND_OUTER_FEATURES_JOB_DONE; date

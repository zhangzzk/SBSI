#!/bin/bash
#SBATCH --job-name=ab_l10an
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_l10an_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"
OUT=results/anchorblend_g005_local10_c200-299.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_local10.py \
  --all-response results/anchorblend_g005_response_v22_c100-299.feather \
  --local-response results/anchorblend_g005_local10_response_v22_c200-299.feather \
  --case-min 200 --case-max 299 --output "$OUT"
echo ANCHORBLEND_LOCAL10_ANALYSIS_JOB_DONE; date

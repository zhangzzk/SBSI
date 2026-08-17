#!/bin/bash
#SBATCH --job-name=ab_fixedana
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_fixedana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_fixedana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
OUT=$ROOT/results/anchorblend_fixed_position_v22_c200-299.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchor_fixed_positions.py \
  --input-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_fixedpos_c200-299 \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --output "$OUT"
echo ANCHOR_FIXED_POSITION_ANALYSIS_JOB_DONE; date

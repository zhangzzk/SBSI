#!/bin/bash
#SBATCH --job-name=abamp_dom
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abamp_dom_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

OUT=results/anchorblend_amplitude_by_dominance_v22_c400-499.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/analyze_anchor_amplitude_by_dominance.py \
  --g002 results/anchorblend_g002_response_v22_c400-499.feather \
  --g005 /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather \
  --dominance results/anchorblend_response_dominance_v22_c400-499.feather \
  --ratio-threshold 20 \
  --case-min 400 --case-max 499 \
  --output "$OUT"

echo ANCHOR_AMPLITUDE_BY_DOMINANCE_JOB_DONE; date

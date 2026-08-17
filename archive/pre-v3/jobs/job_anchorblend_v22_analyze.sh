#!/bin/bash
#SBATCH --job-name=ab_v22ana
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_v22ana_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

OUT=results/anchorblend_g005_v22_diagnostic_c0-299.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_v22_diagnostic.py \
  --new results/anchorblend_g005_response_v22_c0-99.feather \
        results/anchorblend_g005_response_v22_c100-299.feather \
  --original results/anchorblend_g005_response.feather \
             results/anchorblend_g005_response_ext.feather \
  --split-case 100 --output "$OUT"

echo ANCHORBLEND_V22_ANALYZE_DONE; date

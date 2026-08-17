#!/bin/bash
#SBATCH --job-name=ab_ratiocut
#SBATCH --time=00:10:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_ratiocut_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_ratiocut_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
DOMINANCE=$ROOT/results/anchorblend_response_dominance_v22_c400-499.feather
COMMON=$ROOT/results/anchorblend_dominant_pair_domain_v22_c400-499.feather
OUTPUT=$ROOT/results/anchorblend_response_ratio_cut_scan_v22_c400-499.json
test -s "$DOMINANCE"
test -s "$COMMON"
test ! -e "$OUTPUT" || { echo "REFUSING existing $OUTPUT"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/scan_anchorblend_response_ratio_cut.py \
  --dominance-table "$DOMINANCE" --common-table "$COMMON" \
  --primary-threshold 20 \
  --thresholds 5 10 15 20 25 30 40 50 75 100 \
  --development-max 449 --output-json "$OUTPUT"
echo ANCHORBLEND_RESPONSE_RATIO_CUT_SCAN_JOB_DONE
date

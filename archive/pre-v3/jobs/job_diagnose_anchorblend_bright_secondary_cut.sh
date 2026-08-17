#!/bin/bash
#SBATCH --job-name=ab_brightcut
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_brightcut_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_brightcut_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RESPONSE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
DOMINANT=$ROOT/results/anchorblend_dominant_pair_domain_v22_c400-499.feather
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
OUTPUT=$ROOT/results/anchorblend_true_bright_secondary_cut5_v22_c400-499.json
test ! -e "$OUTPUT" || { echo "REFUSING existing $OUTPUT"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/diagnose_anchorblend_bright_secondary_cut.py \
  --response "$RESPONSE" --dominant-table "$DOMINANT" \
  --manifest-dir "$MANIFEST" --reference-base "$REFERENCE" \
  --ratio-threshold 5 --case-min 400 --case-max 499 --development-max 449 \
  --output-json "$OUTPUT"
echo ANCHORBLEND_BRIGHT_SECONDARY_CUT_JOB_DONE
date

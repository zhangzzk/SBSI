#!/bin/bash
#SBATCH --job-name=ab_dom_pair
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_dom_pair_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_dom_pair_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RESPONSE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
DESIGN=$MANIFEST/design.json
OUT_FEATHER=$ROOT/results/anchorblend_dominant_pair_domain_v22_c400-499.feather
OUT_JSON=$ROOT/results/anchorblend_dominant_pair_domain_v22_c400-499.json
OUT_MD=$ROOT/results/anchorblend_dominant_pair_domain_v22_c400-499.md
for output in "$OUT_FEATHER" "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/characterize_anchorblend_dominant_pairs.py \
  --response "$RESPONSE" --manifest-dir "$MANIFEST" \
  --reference-base "$REFERENCE" --design-json "$DESIGN" \
  --case-min 400 --case-max 499 --development-max 449 \
  --dominance-threshold 0.7752772106835227 \
  --output-feather "$OUT_FEATHER" --output-json "$OUT_JSON" --output-md "$OUT_MD"
echo ANCHORBLEND_DOMINANT_PAIR_DOMAIN_JOB_DONE
date

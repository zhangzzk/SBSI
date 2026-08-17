#!/bin/bash
#SBATCH --job-name=ab_respdom
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_respdom_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_respdom_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
OUT_FEATHER=$ROOT/results/anchorblend_response_dominance_v22_c400-499.feather
OUT_JSON=$ROOT/results/anchorblend_response_dominance_v22_c400-499.json
OUT_MD=$ROOT/results/anchorblend_response_dominance_v22_c400-499.md
for output in "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
EXTRA=()
if test -e "$OUT_FEATHER"; then
  test -s "$OUT_FEATHER" || { echo "REFUSING empty $OUT_FEATHER"; exit 1; }
  EXTRA+=(--reuse-feather)
fi
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/localize_anchorblend_response_dominance.py \
  --manifest-dir "$MANIFEST" --reference-base "$REFERENCE" \
  --tag lsst_r_extnbr_v22 --case-min 400 --case-max 499 --development-max 449 \
  --dominance-threshold 0.7752772106835227 \
  --output-feather "$OUT_FEATHER" --output-json "$OUT_JSON" --output-md "$OUT_MD" \
  "${EXTRA[@]}"
echo ANCHORBLEND_RESPONSE_DOMINANCE_LOCALIZATION_JOB_DONE
date

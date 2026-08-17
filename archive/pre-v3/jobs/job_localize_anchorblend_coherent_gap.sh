#!/bin/bash
#SBATCH --job-name=abloc_gap
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abloc_gap_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abloc_gap_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RESPONSE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchorblend_oneactive_orthogonal_c400-499
JSON=$ROOT/results/anchorblend_coherent_localization_v22_c400-499.json
CSV=$ROOT/results/anchorblend_coherent_localization_v22_c400-499.csv
REPORT=$ROOT/results/anchorblend_coherent_localization_v22_c400-499.md
for output in "$JSON" "$CSV" "$REPORT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/localize_anchorblend_coherent_gap.py \
  --response "$RESPONSE" --manifest-dir "$MANIFEST" \
  --case-min 400 --case-max 499 --development-max 449 \
  --output-json "$JSON" --output-csv "$CSV" --output-md "$REPORT"
test -s "$JSON"
test -s "$CSV"
test -s "$REPORT"

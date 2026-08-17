#!/bin/bash
#SBATCH --job-name=ab_biasfeat
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
OUT_FEATHER=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUT_JSON=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899_eda.json
OUT_MD=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899_eda.md
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT_FEATHER" "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"

RESPONSES=()
DOMINANCE=()
DESIGNS=()
for block in 400-499 500-599 600-699 700-799 800-899; do
  RESPONSES+=("results/anchorblend_g002_response_v22_c${block}.feather")
  DOMINANCE+=("results/anchorblend_g002_dominance_renderer_v22_c${block}.feather")
  DESIGNS+=("results/anchorblend_g002_pairs_renderer_v22_c${block}.json")
done
for input in "${RESPONSES[@]}" "${DOMINANCE[@]}" "${DESIGNS[@]}"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done

python -u scripts/build_anchor_bias_features.py \
  --response "${RESPONSES[@]}" --dominance "${DOMINANCE[@]}" \
  --pair-design "${DESIGNS[@]}" --train-max 599 --tune-max 699 \
  --output-feather "$OUT_FEATHER" --output-json "$OUT_JSON" \
  --output-md "$OUT_MD"
echo ANCHOR_BIAS_FEATURES_JOB_DONE
date

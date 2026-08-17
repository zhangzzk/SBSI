#!/bin/bash
#SBATCH --job-name=abtoy_case
#SBATCH --array=700-899%40
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CASE=${SLURM_ARRAY_TASK_ID:?array task required}
MANIFEST=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.feather
OUTDIR=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noiseless_toy_v22_c700-899}
MAX_ANCHORS=${TOY_MAX_ANCHORS:-0}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUTDIR"
ANCHOR=$OUTDIR/anchors_case${CASE}.feather
PAIR=$OUTDIR/pairs_case${CASE}.feather
AUDIT=$OUTDIR/audit_case${CASE}.json
for output in "$ANCHOR" "$PAIR" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
"$PY" -u scripts/run_localized_anchor_noiseless_toy_case.py \
  --manifest "$MANIFEST" \
  --pair-design \
    results/anchorblend_g002_pairs_renderer_v22_c700-799.json \
    results/anchorblend_g002_pairs_renderer_v22_c800-899.json \
  --case "$CASE" --g 0.02 --stamp 48 --max-anchors "$MAX_ANCHORS" \
  --output-anchor "$ANCHOR" --output-pair "$PAIR" --output-json "$AUDIT"
test -s "$ANCHOR"
test -s "$AUDIT"
echo LOCALIZED_ANCHOR_NOISELESS_TOY_CASE_JOB_DONE
date

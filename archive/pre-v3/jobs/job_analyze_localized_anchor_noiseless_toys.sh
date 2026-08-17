#!/bin/bash
#SBATCH --job-name=abtoy_ana
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
MANIFEST=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.feather
INPUT=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noiseless_toy_v22_c700-899}
OUT=$ROOT/results/localized_anchor_noiseless_toy_v22_c700-899
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$OUT.feather" "${OUT}_pairs.feather" "$OUT.json" "$OUT.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/analyze_localized_anchor_noiseless_toys.py \
  --manifest "$MANIFEST" --input-dir "$INPUT" --cases $(seq 700 899) \
  --output-anchor "$OUT.feather" --output-pair "${OUT}_pairs.feather" \
  --output-json "$OUT.json" --output-md "$OUT.md" \
  --min-success-fraction 0.9
test -s "$OUT.json"
test -s "$OUT.md"
echo LOCALIZED_ANCHOR_NOISELESS_TOY_ANALYSIS_JOB_DONE
date

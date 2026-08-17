#!/bin/bash
#SBATCH --job-name=toy_nc_ana
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
INPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/legacy_toy_no_context_exact_v1
OUT=$ROOT/results/legacy_toy_no_context_exact_v1
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
cd "$ROOT"
for output in "$OUT.csv" "$OUT.json" "$OUT.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/analyze_legacy_toy_no_context.py \
  --input-root "$INPUT" \
  --output-csv "$OUT.csv" --output-json "$OUT.json" --output-md "$OUT.md"
for output in "$OUT.csv" "$OUT.json" "$OUT.md"; do
  test -s "$output"
done
echo LEGACY_TOY_NO_CONTEXT_ANALYSIS_JOB_DONE
date

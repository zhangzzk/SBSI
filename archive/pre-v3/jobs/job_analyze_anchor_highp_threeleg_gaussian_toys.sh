#!/bin/bash
#SBATCH --job-name=ab3g_ana
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
MANIFEST=$ROOT/results/anchor_highp_threeleg_gaussian_manifest_v22_c700-899.feather
INDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_highp_threeleg_gaussian_v22_g002_r200
GCHECK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_highp_threeleg_gaussian_v22_g001_r200
OUT=$ROOT/results/anchor_highp_threeleg_gaussian_v22_g002_r200
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in \
  "$OUT.feather" "$OUT.pairs.feather" "$OUT.summary.csv" \
  "$OUT.gcheck.csv" "$OUT.json" "$OUT.md" "$OUT.png" "$OUT.pdf"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/analyze_anchor_highp_threeleg_gaussian_toys.py \
  --templates "$MANIFEST" --input-dir "$INDIR" --nreal 200 --g 0.02 \
  --gcheck-dir "$GCHECK" --gcheck-g 0.01 \
  --min-success-fraction 0.8 --n-bootstrap 20000 --bootstrap-seed 20260814 \
  --output-cells "$OUT.feather" --output-pairs "$OUT.pairs.feather" \
  --output-summary "$OUT.summary.csv" --output-sensitivity "$OUT.gcheck.csv" \
  --output-json "$OUT.json" --output-md "$OUT.md" \
  --output-png "$OUT.png" --output-pdf "$OUT.pdf"
for output in \
  "$OUT.feather" "$OUT.pairs.feather" "$OUT.summary.csv" \
  "$OUT.gcheck.csv" "$OUT.json" "$OUT.md" "$OUT.png" "$OUT.pdf"; do
  test -s "$output"
done
echo ANCHOR_HIGHP_THREELEG_GAUSSIAN_ANALYSIS_JOB_DONE
date

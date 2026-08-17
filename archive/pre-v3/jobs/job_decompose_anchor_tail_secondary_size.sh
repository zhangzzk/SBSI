#!/bin/bash
#SBATCH --job-name=abtailre
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
CURVES=$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final_curves.csv
OUT=$ROOT/results/anchorblend_panelE_tail_secondary_size_cut0p4_c700-899
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for input in "$FEATURES" "$CURVES"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in json md csv; do
  test ! -e "$OUT.$suffix" || { echo "REFUSING existing $OUT.$suffix"; exit 1; }
done
cd "$ROOT"

"$PY" -u scripts/decompose_anchor_tail_secondary_size.py \
  --features "$FEATURES" --parent-curves "$CURVES" \
  --case-min 700 --case-max 899 --size-cut 0.4 \
  --output-prefix "$OUT"
echo ANCHOR_TAIL_SECONDARY_SIZE_DECOMPOSITION_JOB_DONE
date

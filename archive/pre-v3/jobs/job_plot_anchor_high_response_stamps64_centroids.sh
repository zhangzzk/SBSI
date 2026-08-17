#!/bin/bash
#SBATCH --job-name=anchor_stamps64
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
BASE1=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c700-799
BASE2=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c800-899
OUT=$ROOT/results/anchorblend_g002_high_response_gt0p1_stamps64_centroids_c700-899
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for input in "$FEATURES" "$BASE1" "$BASE2"; do
  test -e "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in csv json md npz pdf png; do
  test ! -e "$OUT.$suffix" || {
    echo "REFUSING existing $OUT.$suffix"; exit 1; }
done

cd "$ROOT"
"$PY" -u scripts/plot_anchor_high_response_stamps.py \
  --features "$FEATURES" \
  --base "$BASE1" \
  --base "$BASE2" \
  --case-min 700 \
  --case-max 899 \
  --threshold 0.1 \
  --n-select 64 \
  --shear-leg 0.02 \
  --stamp-size 48 \
  --pixel-scale 0.2 \
  --show-detected-centroid \
  --output-prefix "$OUT"
echo ANCHOR_HIGH_RESPONSE_STAMPS64_JOB_DONE

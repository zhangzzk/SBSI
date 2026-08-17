#!/bin/bash
#SBATCH --job-name=ab_dalpha
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE_ROOT=/project/ls-gruen/users/zekang.zhang
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_distance_exponent_scan_v2
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
STEM=$ROOT/results/anchor_distance_exponent_scan_v22_c400-899_v2

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

mkdir -p "$RUN"
python -u scripts/plot_anchor_distance_exponent_scan.py \
  --features "$FEATURES" --base-root "$BASE_ROOT" \
  --pair-prefix pairs_renderer_v22 --case-min 400 --case-max 899 \
  --shear 0.02 --n-bins 20 \
  --x-quantile-min 0.20 --x-quantile-max 0.98 --hist-bins 32 \
  --table-output "$RUN/anchor_distance_exponent_scan_v22_c400-899.feather" \
  --curve-output "$STEM.curves.csv" \
  --figure-output "$STEM.png" --pdf-output "$STEM.pdf" \
  --summary-output "$STEM.json"

echo ANCHOR_DISTANCE_EXPONENT_SCAN_JOB_DONE
date

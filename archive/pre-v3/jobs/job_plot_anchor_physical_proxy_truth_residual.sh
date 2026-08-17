#!/bin/bash
#SBATCH --job-name=ab_physbin
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_physical_proxy_truth_residual_v1
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
STEM=$ROOT/results/anchor_physical_proxy_truth_residual_v22_c400-899

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

mkdir -p "$RUN"
python -u scripts/plot_anchor_physical_proxy_truth_residual.py \
  --features "$FEATURES" --base-root "$BASE_ROOT" \
  --pair-prefix pairs_renderer_v22 --case-min 400 --case-max 899 \
  --shear 0.02 --n-bins 20 \
  --table-output "$RUN/anchor_physical_proxy_truth_residual_v22_c400-899.feather" \
  --curve-output "$STEM.curves.csv" \
  --figure-output "$STEM.png" --pdf-output "$STEM.pdf" \
  --summary-output "$STEM.json"

echo ANCHOR_PHYSICAL_PROXY_TRUTH_RESIDUAL_JOB_DONE
date

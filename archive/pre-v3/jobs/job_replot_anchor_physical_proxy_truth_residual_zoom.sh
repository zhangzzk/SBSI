#!/bin/bash
#SBATCH --job-name=ab_physzoom
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_physical_proxy_truth_residual_v1
TABLE=$RUN/anchor_physical_proxy_truth_residual_v22_c400-899.feather
CURVES=$ROOT/results/anchor_physical_proxy_truth_residual_v22_c400-899.curves.csv
STEM=$ROOT/results/anchor_physical_proxy_truth_residual_v22_c400-899_xzoom_q20-98_hist

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

python -u scripts/replot_anchor_physical_proxy_truth_residual_zoom.py \
  --table "$TABLE" --curves "$CURVES" \
  --x-quantile-min 0.20 --x-quantile-max 0.98 \
  --figure-output "$STEM.png" --pdf-output "$STEM.pdf"

echo ANCHOR_PHYSICAL_PROXY_TRUTH_RESIDUAL_ZOOM_JOB_DONE
date

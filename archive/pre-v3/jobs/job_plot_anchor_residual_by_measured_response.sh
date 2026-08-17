#!/bin/bash
#SBATCH --job-name=ab_truth_cond
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/plot_anchor_residual_by_measured_response.py \
  --score-dir "$RUN/anchor_scores_c400_899" \
  --anchor-features "$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather" \
  --output-prefix "$ROOT/results/v22_crossfit_anchor_residual_by_measured_response_c400-899_v2"
echo ANCHOR_RESIDUAL_BY_MEASURED_RESPONSE_JOB_DONE
date

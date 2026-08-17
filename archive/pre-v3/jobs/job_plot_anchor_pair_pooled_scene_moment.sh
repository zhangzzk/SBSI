#!/bin/bash
#SBATCH --job-name=poolAnchorPlot
#SBATCH --time=00:45:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_pair_pooled_scene_moment_case100_100_v1
FEATURES=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUTPUT=$ROOT/results/anchor_pair_pooled_scene_moment_c700-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -u scripts/plot_anchor_pair_pooled_scene_moment.py \
  --anchor-features "$FEATURES" \
  --score-dir "$RUN/anchor_scores_c400_899" \
  --pooled-summary "$RUN/summary.json" \
  --output-stem "$OUTPUT"
echo "PLOT_ANCHOR_PAIR_POOLED_SCENE_JOB_DONE"

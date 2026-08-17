#!/bin/bash
#SBATCH --job-name=ab_seqown_plot
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
NEW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3/anchor_scores_c400_899
EXTRA=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_ownscene_optuna50_v1/anchor_scores_c400_899
OLD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/anchor_scores
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$EXTRA/case${case}.feather"
  test -s "$EXTRA/case${case}.json"
done
python -u scripts/plot_anchor_calibration_common_vs_own.py \
  --new-score-dir "$NEW" --old-score-dir "$OLD" \
  --extra-score-dir "$EXTRA" \
  --anchor-features "$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather" \
  --scene-cache "$SCENE" \
  --output-prefix "$ROOT/results/v22_anchor_calibration_common_vs_own_with_corrected_own_c400-899_v2"
echo ANCHOR_CROSSFIT_SEQUENTIAL_OWNSCENE_PLOT_DONE
date

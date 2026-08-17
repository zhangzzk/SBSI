#!/bin/bash
#SBATCH --job-name=ab_full_eval
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_fullneighbour_fixed_v1
NEW=$RUN/anchor_scores_c400_899
PREVIOUS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_ownscene_optuna50_v1/anchor_scores_c400_899
TRUTH=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUTPUT=$ROOT/results/v22_fullneighbour_fixed_anchor_transfer_c400-899_v2.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$NEW/case${case}.feather"
  test -s "$NEW/case${case}.json"
  test -s "$PREVIOUS/case${case}.feather"
done
test -s "$TRUTH"
python -u scripts/evaluate_anchor_fullneighbour_transfer.py \
  --new-score-dir "$NEW" \
  --previous-score-dir "$PREVIOUS" \
  --anchor-features "$TRUTH" \
  --output-json "$OUTPUT"
echo "EVALUATE_ANCHOR_FULLNEIGHBOUR_TRANSFER_JOB_DONE"
date

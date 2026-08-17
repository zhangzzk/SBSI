#!/bin/bash
#SBATCH --job-name=ab_newold_eval
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_residual_v1
NEW=$RUN/anchor_scores_c400_899
EARLY=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/anchor_scores
TRUTH=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUTPUT=$ROOT/results/v22_newbase_oldway_anchor_transfer_c400-899.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$NEW/case${case}.feather"
  test -s "$NEW/case${case}.json"
  test -s "$EARLY/case${case}.feather"
done
python -u scripts/evaluate_anchor_newbase_oldway.py \
  --new-score-dir "$NEW" \
  --early-score-dir "$EARLY" \
  --anchor-features "$TRUTH" \
  --output-json "$OUTPUT"
echo "EVALUATE_ANCHOR_NEWBASE_OLDWAY_JOB_DONE"
date

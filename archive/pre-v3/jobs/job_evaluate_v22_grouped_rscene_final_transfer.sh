#!/bin/bash
#SBATCH --job-name=v22grptrans
#SBATCH --time=02:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/run
ANCHOR_SCORE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/anchor_scores
ANCHOR_FEATURE=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUT=$ROOT/results/v22_grouped_rscene_final_transfer_v1_c20-39_anchor_c400-899
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$ANCHOR_SCORE/case${case}.feather"
done
python -u scripts/evaluate_v22_grouped_rscene_final_transfer.py \
  --source-cache "$SOURCE" --scene-cache "$SCENE" \
  --model-summary "$RUN/final.summary.json" \
  --anchor-score-dir "$ANCHOR_SCORE" --anchor-features "$ANCHOR_FEATURE" \
  --output-prefix "$OUT"
echo V22_GROUPED_RSCENE_FINAL_TRANSFER_JOB_DONE
date

#!/bin/bash
#SBATCH --job-name=v22q2eval
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
OLD_SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1
OLD_RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_proxy_qd2_correction_v1
ANCHOR_FEATURE=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUT=$ROOT/results/v22_proxy_qd2_correction_transfer_c20-39_anchor_c400-899_v2
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$OLD_RUN/anchor_scores/case${case}.feather"
  test -s "$RUN/anchor_scores/case${case}.feather"
done
python -u scripts/evaluate_v22_proxy_correction_transfer.py \
  --source-cache "$SOURCE" --old-scene-cache "$OLD_SCENE" \
  --full-neighbour-cache "$FULL" \
  --old-model-summary "$OLD_RUN/run/final.summary.json" \
  --proxy-model-summary "$RUN/model.summary.json" \
  --old-anchor-score-dir "$OLD_RUN/anchor_scores" \
  --proxy-anchor-score-dir "$RUN/anchor_scores" \
  --anchor-features "$ANCHOR_FEATURE" --output-prefix "$OUT"
echo V22_PROXY_QD2_CORRECTION_EVALUATION_JOB_DONE
date

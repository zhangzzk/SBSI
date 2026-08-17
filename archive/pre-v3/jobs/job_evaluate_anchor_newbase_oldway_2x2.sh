#!/bin/bash
#SBATCH --job-name=ab_2x2_eval
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
MISSING=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_2x2_ablation_v1/anchor_scores_c400_899
ORIG_HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_residual_v1
ORIG_FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_40_199_v1
NEWSPLIT_HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_halfneighbour_0_159_v1
NEWSPLIT_FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_160_40_v2
TRUTH=$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather
OUTPUT=$ROOT/results/v22_newbase_oldway_2x2_anchor_transfer_c400-899.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

for case in $(seq 400 899); do
  test -s "$MISSING/case${case}.feather"
done
python -u scripts/evaluate_anchor_newbase_oldway_2x2.py \
  --missing-score-dir "$MISSING" \
  --orig-half-score-dir "$ORIG_HALF/anchor_scores_c400_899" \
  --newsplit-full-score-dir "$NEWSPLIT_FULL/anchor_scores_c400_899" \
  --anchor-features "$TRUTH" \
  --orig-half-summary "$ORIG_HALF/summary.json" \
  --orig-full-summary "$ORIG_FULL/summary.json" \
  --newsplit-half-summary "$NEWSPLIT_HALF/summary.json" \
  --newsplit-full-summary "$NEWSPLIT_FULL/summary.json" \
  --output-json "$OUTPUT"
echo "EVALUATE_ANCHOR_NEWBASE_OLDWAY_2X2_JOB_DONE"
date

#!/bin/bash
#SBATCH --job-name=ab_2x2
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-99
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
ORIG_FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_40_199_v1
NEWSPLIT_HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_halfneighbour_0_159_v1
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_2x2_ablation_v1/anchor_scores_c400_899
START=$((400 + 5 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 4))

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$ORIG_FULL/summary.json"
test -s "$NEWSPLIT_HALF/summary.json"
mkdir -p "$OUTPUT"
for case in $(seq "$START" "$STOP"); do
  python -u scripts/score_anchor_crossfit_sequential.py \
    --stack-summary "orig_full=$ORIG_FULL/summary.json" \
    --stack-summary "newsplit_half=$NEWSPLIT_HALF/summary.json" \
    --case "$case" --output-dir "$OUTPUT"
done
echo "ANCHOR_NEWBASE_OLDWAY_2X2_ARRAY_DONE cases=$START-$STOP"
date

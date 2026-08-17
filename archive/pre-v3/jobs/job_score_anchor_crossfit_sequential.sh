#!/bin/bash
#SBATCH --job-name=ab_seq2
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
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3
OUTPUT=$RUN_ROOT/anchor_scores_c400_899
START=$((400 + 5 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 4))
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$RUN_ROOT/pair/summary.json"
test -s "$RUN_ROOT/pair_scene/summary.json"
mkdir -p "$OUTPUT"
for case in $(seq "$START" "$STOP"); do
  python -u scripts/score_anchor_crossfit_sequential.py \
    --pair-summary "$RUN_ROOT/pair/summary.json" \
    --pair-scene-summary "$RUN_ROOT/pair_scene/summary.json" \
    --case "$case" --output-dir "$OUTPUT"
done
echo "ANCHOR_CROSSFIT_SEQUENTIAL_ARRAY_DONE cases=$START-$STOP"
date

#!/bin/bash
#SBATCH --job-name=poolScene100
#SBATCH --time=04:00:00
#SBATCH --mem=280G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_case100_100_v1
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_pair_pooled_scene_moment_case100_100_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE/metadata.json"
test -s "$FULL/metadata.json"
test -s "$BASE/summary.json"
python -u scripts/train_case100_100_pair_pooled_scene_moment.py \
  --source-cache "$SOURCE" \
  --full-neighbour-cache "$FULL" \
  --source-stack-summary "$BASE/summary.json" \
  --output-dir "$OUTPUT"
echo "TRAIN_CASE100_100_PAIR_POOLED_SCENE_MOMENT_JOB_DONE"
date

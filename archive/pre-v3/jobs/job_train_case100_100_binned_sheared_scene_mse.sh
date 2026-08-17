#!/bin/bash
#SBATCH --job-name=binScene100
#SBATCH --time=04:00:00
#SBATCH --mem=250G
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
SOURCE_STACK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_case100_100_v1
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_case100_100_binned_sheared_scene_mse_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

test -s "$SOURCE/metadata.json"
test -s "$FULL/metadata.json"
test -s "$SOURCE_STACK/summary.json"
python -u scripts/train_case100_100_binned_sheared_scene_mse.py \
  --source-cache "$SOURCE" \
  --full-neighbour-cache "$FULL" \
  --source-stack-summary "$SOURCE_STACK/summary.json" \
  --output-dir "$OUTPUT"
echo "TRAIN_CASE100_100_BINNED_SHEARED_SCENE_MSE_JOB_DONE"
date

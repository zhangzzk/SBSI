#!/bin/bash
#SBATCH --job-name=rbl100x100
#SBATCH --time=04:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1
PARAMS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_fullneighbour_fixed_v1/summary.json
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_newbase_oldway_fullneighbour_case100_100_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$OUTPUT"
test -s "$SOURCE/metadata.json"
test -s "$FULL/metadata.json"
test -s "$PARAMS"
nvidia-smi -L
python -u scripts/train_newbase_oldway_fullneighbour_case100_100.py \
  --source-cache "$SOURCE" \
  --full-neighbour-cache "$FULL" \
  --base-parameter-summary "$PARAMS" \
  --output-dir "$OUTPUT"
echo "TRAIN_NEWBASE_OLDWAY_FULLNEIGHBOUR_CASE100_100_JOB_DONE"
date

#!/bin/bash
#SBATCH --job-name=rblfullfix
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
FULL=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_fullneighbour_context_v1
PARAMS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_ownscene_optuna50_v1/summary.json
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_fullneighbour_fixed_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$OUTPUT"
test -s "$SOURCE/metadata.json"
test -s "$SCENE/metadata.json"
test -s "$FULL/metadata.json"
test -s "$PARAMS"
nvidia-smi -L
python -u scripts/train_crossfit_sequential_fullneighbour_fixed.py \
  --source-cache "$SOURCE" \
  --scene-cache "$SCENE" \
  --full-neighbour-cache "$FULL" \
  --parameter-summary "$PARAMS" \
  --output-dir "$OUTPUT"
echo "TRAIN_CROSSFIT_SEQUENTIAL_FULLNEIGHBOUR_FIXED_JOB_DONE"
date

#!/bin/bash
#SBATCH --job-name=rblcorown50
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
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
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_ownscene_optuna50_v1
SEED_PAIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3/pair/summary.json
SEED_PAIR_SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_crossfit_sequential_optuna50_v3/pair_scene/summary.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$OUTPUT"
test -s "$SOURCE/metadata.json"
test -s "$SCENE/metadata.json"
test -s "$SEED_PAIR"
test -s "$SEED_PAIR_SCENE"
nvidia-smi -L
python -u scripts/tune_crossfit_sequential_ownscene_optuna.py \
  --source-cache "$SOURCE" \
  --scene-cache "$SCENE" \
  --n-trials 50 \
  --seed-summary "$SEED_PAIR" \
  --seed-summary "$SEED_PAIR_SCENE" \
  --output-dir "$OUTPUT"
echo "CROSSFIT_SEQUENTIAL_OWNSCENE_JOB_DONE"
date

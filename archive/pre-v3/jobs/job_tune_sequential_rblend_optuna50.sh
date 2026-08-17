#!/bin/bash
#SBATCH --job-name=rblseq50
#SBATCH --time=2-00:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-1%2
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
SOURCE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1/cache
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_sequential_optuna50_v1
OBJECTIVES=(pair pair_scene)
OBJECTIVE=${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$RUN_ROOT/$OBJECTIVE"
test -s "$SOURCE/metadata.json"
test -s "$SCENE/metadata.json"
nvidia-smi -L
python -u scripts/tune_sequential_rblend_optuna.py \
  --source-cache "$SOURCE" \
  --scene-cache "$SCENE" \
  --objective "$OBJECTIVE" \
  --n-trials 50 \
  --output-dir "$RUN_ROOT/$OBJECTIVE"
echo "SEQUENTIAL_RBLEND_OPTUNA_JOB_DONE objective=$OBJECTIVE"
date

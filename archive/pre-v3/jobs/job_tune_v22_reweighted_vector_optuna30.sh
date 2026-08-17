#!/bin/bash
#SBATCH --job-name=v22rwv40
#SBATCH --time=2-00:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter,cip
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
OUTPUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XGB_DEVICE=cuda
cd "$ROOT"

mkdir -p "$OUTPUT"
test -s "$CACHE/metadata.json"
nvidia-smi -L
python -u scripts/tune_v22_reweighted_vector_optuna.py \
  --cache "$CACHE" \
  --n-trials 30 \
  --output-dir "$OUTPUT"
echo V22_REWEIGHTED_VECTOR_OPTUNA_JOB_DONE
date

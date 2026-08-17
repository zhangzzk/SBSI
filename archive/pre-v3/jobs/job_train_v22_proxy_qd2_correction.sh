#!/bin/bash
#SBATCH --job-name=v22q2train
#SBATCH --time=04:00:00
#SBATCH --mem=128G
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
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_proxy_qd2_correction_v1
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"

mkdir -p "$RUN"
python -u scripts/train_v22_proxy_correction.py \
  --source-cache "$SOURCE" --full-neighbour-cache "$FULL" \
  --train-case-min 40 --train-case-max 199 --edge-case-max 159 \
  --n-proxy-bins 20 --trees 180 \
  --output-model "$RUN/model.json" \
  --output-summary "$RUN/model.summary.json"
echo V22_PROXY_QD2_CORRECTION_TRAIN_JOB_DONE
date

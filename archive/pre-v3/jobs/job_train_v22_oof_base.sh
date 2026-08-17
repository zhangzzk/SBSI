#!/bin/bash
#SBATCH --job-name=v22oof_base
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --array=0-3%4
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22oof_base_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22oof_base_%A_%a.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/run
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd "$ROOT"
"$PY" -u scripts/train_v22_oof_learning.py base-fold \
  --cache "$CACHE" --output-dir "$RUN" --fold "$SLURM_ARRAY_TASK_ID"


#!/bin/bash
#SBATCH --job-name=v22oof_cmerge
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22oof_cmerge_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22oof_cmerge_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/run
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/train_v22_oof_learning.py merge-bias \
  --cache "$CACHE" --output-dir "$RUN"


#!/bin/bash
#SBATCH --job-name=v22oof_eval
#SBATCH --time=01:00:00
#SBATCH --mem=72G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22oof_eval_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22oof_eval_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/run
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/evaluate_v22_oof_learning.py \
  --cache "$CACHE" --run-dir "$RUN" \
  --output-prefix results/v22_oof_conditional_learning_c0-39


#!/bin/bash
#SBATCH --job-name=v22oof_cache
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22oof_cache_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22oof_cache_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_oof_learning_v1/cache
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/prepare_v22_oof_learning_cache.py \
  --config configs/fs2_lsst_r_extnbr_v22.yaml \
  --output-dir "$CACHE"


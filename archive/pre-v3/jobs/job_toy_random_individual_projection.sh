#!/bin/bash
#SBATCH --job-name=rind_toy
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/toy_random_individual_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/toy_random_individual_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT="$ROOT/results/toy_random_individual_projection_noiseless.json"
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

test ! -e "$OUT"
"$PY" -u scripts/toy_random_individual_projection.py \
  --g 0.05 \
  --n-draw 256 \
  --n-angles 32 \
  --sky 0 \
  --output-json "$OUT"
test -s "$OUT"

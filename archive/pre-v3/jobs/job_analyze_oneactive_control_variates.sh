#!/bin/bash
#SBATCH --job-name=ab1n_cv
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_cv_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_cv_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
INPUT=$ROOT/results/anchorblend_oneactive_orthogonal_v22_c400-499.feather
OUTPUT=$ROOT/results/anchorblend_oneactive_control_variates_v22_c400-499.json
test ! -e "$OUTPUT" || { echo "REFUSING existing $OUTPUT"; exit 1; }
cd "$ROOT"
"$PY" -u scripts/analyze_oneactive_control_variates.py \
  --input "$INPUT" --output "$OUTPUT" --n-bootstrap 2000
echo ONEACTIVE_CONTROL_VARIATES_JOB_DONE

#!/bin/bash
#SBATCH --job-name=hs_ncount
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_ncount_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_ncount_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"
"$PY" -u scripts/analyze_halfshear_neighbour_count_slope.py \
  --table /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_vector_closure_v22_training_c40-199.feather \
  --split-case 120 \
  --output results/halfshear_neighbour_count_slope_v22_training_c40-199.json

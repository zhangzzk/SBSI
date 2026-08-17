#!/bin/bash
#SBATCH --job-name=ab_indana
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchorblend_independent_response.py \
  --independent results/anchorblend_independent_local10_response_v22_c200-299.feather \
  --coherent results/anchorblend_g005_response_v22_c100-299.feather \
  --output results/anchorblend_independent_vs_coherent_v22_c200-299.json
echo ANCHORBLEND_INDEPENDENT_ANALYSIS_JOB_DONE
date

#!/bin/bash
#SBATCH --job-name=ab_indgroupg0an
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indgroupg0an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indgroupg0an_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
TAG=lsst_r_extnbr_v22_groupvecg0
IND=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_${TAG}_c200-299.feather
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchorblend_independent_response.py \
  --independent "$IND" \
  --coherent results/anchorblend_g005_response_v22_c100-299.feather \
  --allow-prediction-difference \
  --output results/anchorblend_independent_${TAG}_c200-299.json

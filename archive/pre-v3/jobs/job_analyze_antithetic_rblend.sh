#!/bin/bash
#SBATCH --job-name=anti_rbana
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anti_rbana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/anti_rbana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
cd "$ROOT"
python -u scripts/analyze_antithetic_rblend.py \
  --input-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_antithetic_rblend_c0-99 \
  --output results/rblend_forward_vs_central_g002_v22_c0-99.json


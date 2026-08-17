#!/bin/bash
#SBATCH --job-name=ab_posblocks
#SBATCH --time=00:20:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_posblocks_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_posblocks_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchor_response_validation_blocks.py \
  --old-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c200-399 \
  --fresh-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c400-599 \
  --tag lsst_r_extnbr_v22_rpowposa0065 \
  --output results/anchorblend_response_positive0065_validation_blocks_c200-599.json

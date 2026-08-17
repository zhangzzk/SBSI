#!/bin/bash
#SBATCH --job-name=ab_truthtail
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_truthtail_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_truthtail_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
python -u scripts/audit_anchorblend_truth_tail.py \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --response results/anchorblend_g005_response_v22_c300-399.feather \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_combined_c0-399 \
  --tag lsst_r_extnbr_v22_rpowa0065 --case-min 200 --case-max 399 \
  --output results/anchorblend_truth_tail_v22_alpha0065_c200-399.json

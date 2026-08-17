#!/bin/bash
#SBATCH --job-name=ab_caseaudit
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_caseaudit_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_caseaudit_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
python -u scripts/audit_anchorblend_case_structure.py \
  --response results/anchorblend_g005_response_v22_c0-99.feather \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --response results/anchorblend_g005_response_v22_c300-399.feather \
  --output results/anchorblend_case_structure_v22_c0-399.json

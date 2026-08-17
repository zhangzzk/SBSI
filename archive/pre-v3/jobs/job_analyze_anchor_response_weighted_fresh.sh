#!/bin/bash
#SBATCH --job-name=abfr_rpowan
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_rpowan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_rpowan_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_c300-399 \
  --case-min 300 --case-max 399 \
  --output results/anchorblend_response_weighted_v22_fresh_c300-399.json
echo ANCHOR_RESPONSE_WEIGHTED_FRESH_ANALYSIS_DONE

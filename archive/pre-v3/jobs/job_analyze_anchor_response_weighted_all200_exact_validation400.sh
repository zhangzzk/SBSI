#!/bin/bash
#SBATCH --job-name=ab_a200x_v400an
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_a200x_v400an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_a200x_v400an_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TAG=lsst_r_extnbr_v22_rpowa0065_all200exact
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_all200exact_c200-599 \
  --case-min 200 --case-max 599 --tags "$TAG" --previously-inspected \
  --output results/anchorblend_response_weighted_all200exact_v22_validation_c200-599.json
echo ANCHOR_RESPONSE_WEIGHTED_ALL200EXACT_VALIDATION400_ANALYSIS_DONE

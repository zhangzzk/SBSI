#!/bin/bash
#SBATCH --job-name=abold_pos3an
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abold_pos3an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abold_pos3an_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c200-399 \
  --case-min 200 --case-max 399 \
  --previously-inspected \
  --tags lsst_r_extnbr_v22_rpowposa0065 lsst_r_extnbr_v22_rpowposa010 lsst_r_extnbr_v22_rpowposa015 \
  --output results/anchorblend_response_positive_weighted_v22_retrospective_c200-399.json
echo ANCHOR_RESPONSE_POSITIVE_WEIGHTED_RETROSPECTIVE_ANALYSIS_DONE

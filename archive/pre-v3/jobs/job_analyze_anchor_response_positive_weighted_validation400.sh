#!/bin/bash
#SBATCH --job-name=ab_posv400
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_posv400_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_posv400_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_combined_c200-599
for TAG in lsst_r_extnbr_v22_rpowposa0065 lsst_r_extnbr_v22_rpowposa010 lsst_r_extnbr_v22_rpowposa015; do
  mkdir -p "$PARTS/$TAG"
  for case in $(seq 200 399); do
    ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c200-399/$TAG/case${case}.feather "$PARTS/$TAG/case${case}.feather"
  done
  for case in $(seq 400 599); do
    ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c400-599/$TAG/case${case}.feather "$PARTS/$TAG/case${case}.feather"
  done
done
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root "$PARTS" --case-min 200 --case-max 599 \
  --previously-inspected \
  --tags lsst_r_extnbr_v22_rpowposa0065 lsst_r_extnbr_v22_rpowposa010 lsst_r_extnbr_v22_rpowposa015 \
  --output results/anchorblend_response_positive_weighted_v22_validation_c200-599.json
echo ANCHOR_RESPONSE_POSITIVE_WEIGHTED_VALIDATION400_ANALYSIS_DONE

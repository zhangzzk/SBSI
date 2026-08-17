#!/bin/bash
#SBATCH --job-name=hs_cent_xfer
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_cent_xfer_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_cent_xfer_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/transfer_halfshear_centroid_gap.py \
  --halfshear /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --anchor /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_detection_c200-299.feather \
  --halfshear-target projected_gap \
  --output results/halfshear_centroid_projected_gap_transfer_to_anchor.json
echo HALFSHEAR_CENTROID_GAP_TRANSFER_JOB_DONE
date

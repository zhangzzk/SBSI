#!/bin/bash
#SBATCH --job-name=ab_indfixana
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indfixana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indfixana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
OUT=$ROOT/results/anchorblend_independent_fixed_position_v22_rpowposa0065_c200-299.json
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchor_independent_fixed_positions.py \
  --measurement-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_fixedpos_c200-299 \
  --baseline results/anchorblend_independent_local10_response_v22_c200-299.feather \
  --candidate /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_lsst_r_extnbr_v22_rpowposa0065_c200-299.feather \
  --output "$OUT"
echo ANCHOR_INDEPENDENT_FIXED_POSITION_ANALYSIS_JOB_DONE

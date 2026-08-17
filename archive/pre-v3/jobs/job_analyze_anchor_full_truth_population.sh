#!/bin/bash
#SBATCH --job-name=ab_fullpop
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_fullpop_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_fullpop_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchor_full_truth_population.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --selected-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_fixedpos_c200-299 \
  --missing-dir /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_missing_truthpos_c200-299 \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --table-output /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_full_truthpop_c200-299.feather \
  --output results/anchorblend_full_truth_population_v22_c200-299.json
echo ANCHOR_FULL_TRUTH_POPULATION_JOB_DONE
date

#!/bin/bash
#SBATCH --job-name=ab_rspdiag
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_rspdiag_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_rspdiag_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"
"$PY" -u scripts/diag_anchorblend_resparsification.py \
  --coherent-response results/anchorblend_g005_response_v22_c100-299.feather \
  --outer-features results/anchorblend_outer_features_c200-299.feather \
  --manifest-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_random_local10_c200-299 \
  --case-min 200 --case-max 299 \
  --output results/anchorblend_resparsification_v22_c200-299.json
echo ANCHORBLEND_RESPARSIFICATION_DIAGNOSTIC_JOB_DONE; date

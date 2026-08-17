#!/bin/bash
#SBATCH --job-name=ab_indrpow3an
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-2%3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indrpow3an_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indrpow3an_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=lsst_r_extnbr_v22_rpowa005 ;;
  1) TAG=lsst_r_extnbr_v22_rpowa0065 ;;
  2) TAG=lsst_r_extnbr_v22_rpowa008 ;;
  *) echo "unexpected array task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac
IND=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_${TAG}_c200-299.feather
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_anchorblend_independent_response.py \
  --independent "$IND" \
  --coherent results/anchorblend_g005_response_v22_c100-299.feather \
  --allow-prediction-difference \
  --output "results/anchorblend_independent_${TAG}_c200-299.json"
echo "ANCHOR_INDEPENDENT_WEIGHTED_FINAL_ANALYSIS_DONE tag=$TAG"

#!/bin/bash
#SBATCH --job-name=abrep2_rpowan
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abrep2_rpowan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abrep2_rpowan_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_c400-599 \
  --case-min 400 --case-max 599 \
  --tags lsst_r_extnbr_v22_rpowa0065 \
  --output results/anchorblend_response_weighted_primary_v22_c400-599.json
echo ANCHOR_RESPONSE_WEIGHTED_PRIMARY_REP2_ANALYSIS_DONE

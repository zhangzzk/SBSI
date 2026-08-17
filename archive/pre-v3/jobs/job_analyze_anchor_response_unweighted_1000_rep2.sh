#!/bin/bash
#SBATCH --job-name=abrep2_t1000an
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abrep2_t1000an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abrep2_t1000an_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TAG=lsst_r_extnbr_v22_rpowunwt1000
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_response_weighted_family.py \
  --parts-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_unwt1000_c400-599 \
  --case-min 400 --case-max 599 --tags "$TAG" \
  --output results/anchorblend_response_unweighted_1000_v22_c400-599.json
echo ANCHOR_RESPONSE_UNWEIGHTED_1000_REP2_ANALYSIS_DONE

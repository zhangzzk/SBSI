#!/bin/bash
#SBATCH --job-name=abfr_pairan
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_pairan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_pairan_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_paircal_c300-399
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_pair_vector_calibration.py \
  --input-dir "$PARTS" --case-min 300 --case-max 399 --development-max 299 \
  --fresh-validation \
  --table-output results/anchorblend_pair_vector_calibration_v22_fresh_c300-399.feather \
  --output results/anchorblend_pair_vector_calibration_v22_fresh_c300-399.json
echo ANCHOR_FRESH_PAIR_VECTOR_CALIBRATION_ANALYSIS_DONE
date

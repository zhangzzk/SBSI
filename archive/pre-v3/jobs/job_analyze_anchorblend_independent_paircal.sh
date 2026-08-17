#!/bin/bash
#SBATCH --job-name=ab_indpairan
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indpairan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indpairan_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_paircal_c200-299
TABLE=$ROOT/results/anchorblend_independent_paircal_v22_c200-299.feather
JSON=$ROOT/results/anchorblend_independent_paircal_v22_c200-299.json
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/combine_anchor_independent_fast.py --input-dir "$PARTS" --output "$TABLE"
python -u scripts/analyze_anchorblend_independent_response.py \
  --independent "$TABLE" \
  --coherent results/anchorblend_g005_response_v22_c100-299.feather \
  --allow-prediction-difference --output "$JSON"
echo ANCHORBLEND_INDEPENDENT_PAIRCAL_ANALYSIS_DONE
date

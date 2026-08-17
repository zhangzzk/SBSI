#!/bin/bash
#SBATCH --job-name=ab_cpairan
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_cpairan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_cpairan_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_coherent_paircap64_c200-299
TABLE=$ROOT/results/anchorblend_coherent_paircap64_v22_c200-299.feather
JSON=$ROOT/results/anchorblend_coherent_paircap64_v22_c200-299.json
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
python -u scripts/analyze_anchor_coherent_pair_cap.py \
  --input-dir "$PARTS" --output-table "$TABLE" --output "$JSON"
echo ANCHORBLEND_COHERENT_PAIRCAP_ANALYSIS_DONE
date

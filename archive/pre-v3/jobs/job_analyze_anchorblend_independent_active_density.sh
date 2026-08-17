#!/bin/bash
#SBATCH --job-name=ab_densityana
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_densityana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_densityana_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_halfactive_fast_c200-299
HALF=$ROOT/results/anchorblend_independent_halfactive_fast_v22_c200-299.feather
ALL=$ROOT/results/anchorblend_independent_local10_fast_v22_c200-299.feather
OUT=$ROOT/results/anchorblend_independent_active_density_v22_c200-299.json
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
cd "$ROOT"
if [ ! -s "$HALF" ]; then
  python -u scripts/combine_anchor_independent_fast.py \
    --input-dir "$PARTS" --output "$HALF"
fi
python -u scripts/analyze_anchor_independent_active_density.py \
  --all-active "$ALL" --half-active "$HALF" --output "$OUT"
echo ANCHORBLEND_INDEPENDENT_ACTIVE_DENSITY_ANALYSIS_DONE
date

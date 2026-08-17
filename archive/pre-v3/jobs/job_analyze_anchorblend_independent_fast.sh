#!/bin/bash
#SBATCH --job-name=ab_indfastana
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indfastana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indfastana_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_fast_c200-299
OUT=$ROOT/results/anchorblend_independent_local10_fast_v22_c200-299.feather
JSON=$ROOT/results/anchorblend_independent_vs_coherent_fast_v22_c200-299.json
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/combine_anchor_independent_fast.py \
  --input-dir "$PARTS" --output "$OUT"
python -u scripts/analyze_anchorblend_independent_response.py \
  --independent "$OUT" \
  --coherent results/anchorblend_g005_response_v22_c100-299.feather \
  --output "$JSON"
echo ANCHORBLEND_INDEPENDENT_FAST_ANALYSIS_DONE
date

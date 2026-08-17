#!/bin/bash
#SBATCH --job-name=v22q3c_ana
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3c_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3c_ana_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
TRUTH=$CACHE/v22_constgold_q3_decomp_c40-139_truth.feather
FLOW=$CACHE/v22_constgold_q3_decomp_c40-139_flow.feather
OUT=$ROOT/results/v22_constgold_q3_decomp_c40-139.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/analyze_v22_constgold_bin_decomposition.py \
  --truth "$TRUTH" --flow "$FLOW" --output "$OUT" \
  --plot-prefix "$ROOT/results/v22_constgold_q3_decomp_c40-139"
echo V22_CONSTGOLD_Q3_DECOMP_C100_ANALYZE_DONE

#!/bin/bash
#SBATCH --job-name=v22q3_ana
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3_ana_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
TRUTH=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_q3_decomp_pilot_truth.feather
FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_q3_decomp_pilot_flow.feather
OUT=$ROOT/results/v22_constgold_q3_decomp_pilot.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/analyze_v22_constgold_bin_decomposition.py \
  --truth "$TRUTH" --flow "$FLOW" --output "$OUT" \
  --plot-prefix "$ROOT/results/v22_constgold_q3_decomp_pilot"
echo V22_CONSTGOLD_Q3_DECOMP_PILOT_ANALYZE_DONE

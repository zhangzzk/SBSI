#!/bin/bash
#SBATCH --job-name=v22rk_ana
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rk_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22rk_ana_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
TRUTH=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot_truth.feather
FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_rankpilot_flow.feather
EDGES=$ROOT/results/v22_matched_decomp_pilot.json
OUT=$ROOT/results/v22_matched_decomp_rankpilot.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/analyze_v22_matched_decomposition.py \
  --truth "$TRUTH" --flow "$FLOW" --output "$OUT" --pilot \
  --edges-from "$EDGES" --plot-prefix "$ROOT/results/v22_matched_decomp_rankpilot"
echo V22_MATCHED_RANKPILOT_ANALYZE_DONE

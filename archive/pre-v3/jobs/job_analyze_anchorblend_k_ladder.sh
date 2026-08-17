#!/bin/bash
#SBATCH --job-name=ab_kladan
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_kladan_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"
OUT=results/anchorblend_g005_k_ladder_c0-299.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_k_ladder.py \
  results/anchorblend_g005_k_ladder_c0-99.feather \
  results/anchorblend_g005_k_ladder_c100-299.feather \
  --ks 20 32 48 64 --output "$OUT"
echo ANCHORBLEND_K_LADDER_ANALYSIS_JOB_DONE; date

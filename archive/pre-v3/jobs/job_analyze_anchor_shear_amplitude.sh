#!/bin/bash
#SBATCH --job-name=abgamp_ana
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abgamp_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abgamp_ana_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
OUT=$ROOT/results/anchorblend_shear_amplitude_v22_c400-499.json
cd "$ROOT"
"$PY" -u scripts/analyze_anchor_shear_amplitude.py \
  --g002 results/anchorblend_g002_response_v22_c400-499.feather \
  --g005 /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather \
  --output "$OUT"
echo ANCHOR_SHEAR_AMPLITUDE_ANALYSIS_JOB_DONE

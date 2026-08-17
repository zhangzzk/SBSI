#!/bin/bash
#SBATCH --job-name=ab_g005c200an
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_g005c200an_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_g005c200an_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export G005_CAND_TAG=lsst_r_extnbr_v22_g005_c200
export G005_TRAINING_CASE_DESCRIPTION="all 200 g=0.05 response cases 0--199 (fixed recipe)"
cd "$ROOT"
OUT=results/anchorblend_emu_g005_c200fit_v22_c200-299.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/analyze_anchorblend_g005_emu.py \
  --coherent results/anchorblend_g005_emu_g005_c200fit_c200-299.feather \
  --random results/anchorblend_random_local10_emu_g005_c200fit_c200-299.feather \
  --random results/anchorblend_random_layer1_local10_emu_g005_c200fit_c200-299.feather \
  --random results/anchorblend_random_layer2_local10_emu_g005_c200fit_c200-299.feather \
  --output "$OUT"
echo ANCHORBLEND_G005_C200_ANALYSIS_DONE; date

#!/bin/bash
#SBATCH --job-name=ab_v22iso
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_v22iso_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
NPZ=results/anchorblend_g005_isotonic_v22_c0-299.npz
JSON=results/anchorblend_g005_isotonic_v22_c0-299.json
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$NPZ" ] && [ ! -e "$JSON" ] || { echo "REFUSING existing isotonic output"; exit 1; }
"$PY" -u scripts/fit_anchorblend_isotonic.py \
  results/anchorblend_g005_response_v22_c0-99.feather \
  results/anchorblend_g005_response_v22_c100-299.feather \
  --tag lsst_r_extnbr_v22 --split-case 100 \
  --output-npz "$NPZ" --output-json "$JSON"

echo ANCHORBLEND_V22_ISOTONIC_DONE; date

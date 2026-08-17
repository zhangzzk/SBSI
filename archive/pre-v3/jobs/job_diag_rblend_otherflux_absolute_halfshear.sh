#!/bin/bash
#SBATCH --job-name=rbabsothhs
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbabsothhs_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rbabsothhs_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

"$PY" -u scripts/diag_rblend_otherflux_halfshear.py \
  --scene-lookup results/neighbor_flux_shells_hs_c0-39.feather \
  --case-min 0 --case-max 40 --flux-mode absolute --zero-point 30 \
  --output-prefix results/rblend_otherflux_absolute_halfshear_c0-39

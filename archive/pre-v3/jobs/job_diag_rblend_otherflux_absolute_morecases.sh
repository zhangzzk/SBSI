#!/bin/bash
#SBATCH --job-name=rbabsmore
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbabsmore_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rbabsmore_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CASE_MIN=${CASE_MIN:?set CASE_MIN}
CASE_MAX=${CASE_MAX:?set CASE_MAX}
LABEL=${LABEL:?set LABEL}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

"$PY" -u scripts/diag_rblend_otherflux_halfshear.py \
  --scene-lookup results/neighbor_flux_shells_hs_c0-39.feather \
  --scene-lookup results/neighbor_flux_shells_hs_c40-199.feather \
  --case-min "$CASE_MIN" --case-max "$CASE_MAX" --flux-mode absolute --zero-point 30 \
  --output-prefix "results/rblend_otherflux_absolute_halfshear_${LABEL}"

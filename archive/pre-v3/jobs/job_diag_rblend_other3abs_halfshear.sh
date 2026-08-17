#!/bin/bash
#SBATCH --job-name=rbo3hs
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbo3hs_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rbo3hs_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
OUT=results/rblend_other3abs_halfshear_c0-39
[ ! -e "$OUT.json" ] || { echo "REFUSING to overwrite $OUT.json"; exit 1; }
"$PY" -u scripts/diag_rblend_otherflux_retrain.py \
  --scene-lookup results/neighbor_flux_shells_hs_c0-39.feather \
  --case-min 0 --case-max 40 --zero-point 30 --output-prefix "$OUT"
echo RBLEND_OTHER3ABS_HALFSHEAR_JOB_DONE
date

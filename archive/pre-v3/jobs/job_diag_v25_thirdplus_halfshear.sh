#!/bin/bash
#SBATCH --job-name=v25tpdiag
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v25tpdiag_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SELF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v25_thirdplus_target_c40-199/halfshear_selfresp_v25_thirdplus_target_c40-199.feather
REF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
FLUX=results/neighbor_flux_shells_hs_c40-199.feather
for f in "$SELF" "$REF" "$FLUX"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
for CASE_WINDOW in 40:199 40:99 100:199; do
  IFS=: read -r MIN_CASE MAX_CASE <<< "$CASE_WINDOW"
  echo "=== CASE WINDOW [$MIN_CASE,$MAX_CASE] ==="
  "$PY" -u scripts/diag_v22_halfshear_thirdplus.py \
    --selfresp "$SELF" --reference-selfresp "$REF" --flux-lookup "$FLUX" \
    --min-case "$MIN_CASE" --max-case "$MAX_CASE" \
    --mag-max 25.8 --re-min 0.5 --mag-bins 32 --other-bins 8 --min-cell 100 \
    --max-smd 0.1 --expect-seeds 501 502 503 505 --contrasts-only
done
echo V25_THIRDPLUS_HALFSHEAR_DIAG_DONE; date

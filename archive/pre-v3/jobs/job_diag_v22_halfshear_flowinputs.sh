#!/bin/bash
#SBATCH --job-name=hsv22inputs
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsv22inputs_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SELF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather
FLUX=results/neighbor_flux_shells_hs_c40-199.feather
for f in "$SELF" "$BASE" "$FLUX"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
echo "### V2.2 HALF-SHEAR THIRD-PLUS AT FIXED FLOW INPUTS job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/diag_v22_halfshear_flowinputs.py \
  --selfresp "$SELF" --base-cache "$BASE" --flux-lookup "$FLUX" \
  --mag-max 25.8 --re-min 0.5 --mag-bins 32 --other-bins 4 \
  --top2-bins 8 --far-bins 8 --min-cell 20 \
  --max-smd 0.1
echo HS_V22_FLOWINPUTS_DONE; date

#!/bin/bash
#SBATCH --job-name=v28rb_prep
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v28rb_prep_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
V=$C/v28_rblend
mkdir -p "$V"
HS_LOOKUP=$V/rblend_hs_c40-199.feather
CONST_LOOKUP=$V/rblend_const_c40-139.feather
HS_BASE=$V/halfshear_rblend_base_c40-199_v22domain.feather
if [ ! -e "$HS_LOOKUP" ]; then
  "$PY" -u scripts/compact_key_feature.py \
    --catalogue "$C/det_meas_crowd_conc_g0.0_train_v22domain.feather" \
                "$C/det_meas_crowd_g0.05_val_v22domain.feather" \
    --input-column r_blend --output-column r_blend --min-case 40 --max-case 199 \
    --output "$HS_LOOKUP"
fi
if [ ! -e "$CONST_LOOKUP" ]; then
  "$PY" -u scripts/compact_key_feature.py \
    --catalogue results/blend_lookup_v22_c40-139.feather \
    --input-column R_blend --output-column r_blend --min-case 40 --max-case 139 \
    --output "$CONST_LOOKUP"
fi
if [ ! -e "$HS_BASE" ]; then
  "$PY" -u scripts/augment_catalogue_lookup.py \
    --catalogue "$C/halfshear_base_c40-199_v22domain.feather" \
    --lookup "$HS_LOOKUP" --columns r_blend --drop-unmatched --output "$HS_BASE"
fi
echo V28_RBLEND_PREP_DONE; date

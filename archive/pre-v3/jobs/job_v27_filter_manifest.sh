#!/bin/bash
#SBATCH --job-name=v27manifest
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v27manifest_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
mkdir -p "$C"

TRAIN=$C/det_meas_crowd_conc_g0.0_train_v22domain.feather
TARGET=$C/det_meas_crowd_g0.05_val_v22domain.feather
BASE=$C/halfshear_base_c40-199_v22domain.feather
for spec in \
  "$D/det_meas_crowd_conc_g0.0_train_full.feather|$TRAIN" \
  "$D/det_meas_crowd_g0.05_val_full.feather|$TARGET" \
  "/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather|$BASE"; do
  IFS='|' read -r INPUT OUTPUT <<< "$spec"
  [ -e "$OUTPUT" ] || "$PY" -u scripts/filter_v22_domain_catalogue.py \
    --input "$INPUT" --output "$OUTPUT" --primary-mag-max 25.8 --primary-re-min 0.5
done

HALF_MANIFEST=$C/scene_anchor_manifest_hs_c0-199.feather
[ -e "$HALF_MANIFEST" ] || "$PY" -u scripts/build_scene_anchor_manifest.py \
  --catalogues "$TRAIN" "$TARGET" "$BASE" --output "$HALF_MANIFEST"

CONST_DOMAIN=$C/constant_response_c40-139_v22domain_retry.feather
[ -e "$CONST_DOMAIN" ] || "$PY" -u scripts/filter_v22_domain_catalogue.py \
  --input /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --output "$CONST_DOMAIN" --primary-mag-max 25.8 --primary-re-min 0.5 \
  --min-case 40 --max-case 140
CONST_MANIFEST=$C/scene_anchor_manifest_const_c40-139.feather
[ -e "$CONST_MANIFEST" ] || "$PY" -u scripts/build_scene_anchor_manifest.py \
  --catalogues "$CONST_DOMAIN" --output "$CONST_MANIFEST"
echo V27_FILTER_MANIFEST_DONE; date

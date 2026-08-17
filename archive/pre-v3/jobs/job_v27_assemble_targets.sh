#!/bin/bash
#SBATCH --job-name=v27assemble
#SBATCH --time=08:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v27assemble_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
HALF=$C/scene_features_hs_c0-199.feather
CONST=$C/scene_features_const_c40-139.feather
[ -e "$HALF" ] || "$PY" -u scripts/compact_scene_feature_shards.py \
  --inputs "$C/parts/scene_features_hs_r2_c*.feather" \
  --manifest "$C/scene_anchor_manifest_hs_c0-199.feather" --output "$HALF"
[ -e "$CONST" ] || "$PY" -u scripts/compact_scene_feature_shards.py \
  --inputs "$C/parts/scene_features_const_r2_c*.feather" \
  --manifest "$C/scene_anchor_manifest_const_c40-139.feather" --output "$CONST"

COLS=(logflux_abs_shell_0_0p5 logflux_abs_shell_0p5_1 logflux_abs_shell_1_2 \
      logflux_abs_shell_2_3 logflux_abs_shell_3_5 logflux_abs_shell_5_10 \
      true_blendedness_eq17)
TRAIN_IN=$C/det_meas_crowd_conc_g0.0_train_v22domain.feather
TARGET_IN=$C/det_meas_crowd_g0.05_val_v22domain.feather
BASE_IN=$C/halfshear_base_c40-199_v22domain.feather
TRAIN=$C/det_meas_scenev27_g0.0_train_v22domain.feather
TARGET=$C/det_meas_scenev27_g0.05_val_v22domain.feather
BASE=$C/halfshear_scenev27_base_c40-199_v22domain.feather
for spec in "$TRAIN_IN|$TRAIN" "$TARGET_IN|$TARGET" "$BASE_IN|$BASE"; do
  IFS='|' read -r INPUT OUTPUT <<< "$spec"
  [ -e "$OUTPUT" ] || "$PY" -u scripts/augment_catalogue_lookup.py \
    --catalogue "$INPUT" --lookup "$HALF" --columns "${COLS[@]}" --output "$OUTPUT"
done
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
for ARM in sixshell purity; do
  OUT=$C/response_target_v27_${ARM}.joblib
  [ -e "$OUT" ] || "$PY" -u scripts/build_scene_response_target_v27.py \
    --catalogue "$TARGET" --snc-lookup "$SNC" --arm "$ARM" --output "$OUT"
done
echo V27_ASSEMBLE_TARGETS_DONE; date

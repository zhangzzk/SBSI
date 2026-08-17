#!/bin/bash
#SBATCH --job-name=v26scene
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v26scene_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

LOW=results/neighbor_flux_shells_hs_c0-39.feather
HIGH=results/neighbor_flux_shells_hs_c40-199.feather
CONST=results/neighbor_flux_shells_const_c40-139.feather
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene
HS_LOOKUP=$SCENE/neighbor_scene_hs_c0-199.feather
CONST_LOOKUP=$SCENE/neighbor_scene_const_c40-139.feather
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
TRAIN_IN=$D/det_meas_crowd_conc_g0.0_train_full.feather
TRAIN_OUT=$D/det_meas_crowd_conc_scene_g0.0_train_full.feather
TARGET_IN=$D/det_meas_crowd_g0.05_val_full.feather
TARGET_OUT=$D/det_meas_crowd_scene_g0.05_val_full.feather
BASE_IN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather
BASE_OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v26_scene_c40-199/base_c40-199.feather
mkdir -p "$SCENE" "$(dirname "$BASE_OUT")"
for f in "$LOW" "$HIGH" "$CONST" "$TRAIN_IN" "$TARGET_IN" "$BASE_IN"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done

echo "### V2.6 JOINT TOP-TWO x THIRD-PLUS SCENE PREP job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/compact_neighbor_scene.py \
  --inputs "$LOW" "$HIGH" --output "$HS_LOOKUP"
"$PY" -u scripts/compact_neighbor_scene.py \
  --inputs "$CONST" --output "$CONST_LOOKUP"
for spec in \
  "$TRAIN_IN|$TRAIN_OUT" \
  "$TARGET_IN|$TARGET_OUT" \
  "$BASE_IN|$BASE_OUT"; do
  IFS='|' read -r INPUT OUTPUT <<< "$spec"
  "$PY" -u scripts/augment_catalogue_lookup.py \
    --catalogue "$INPUT" --lookup "$HS_LOOKUP" \
    --columns nbr_flux_top2 nbr_flux_thirdplus --output "$OUTPUT"
done
echo V26_SCENE_PREP_DONE; date

#!/bin/bash
#SBATCH --job-name=v23prep
#SBATCH --time=01:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v23prep_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

LOW=results/neighbor_flux_shells_hs_c0-39.feather
HIGH=results/neighbor_flux_shells_hs_c40-199.feather
CONST=results/neighbor_flux_shells_const_c40-139.feather
SCENE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene
HS_LOOKUP=$SCENE/neighbor_thirdplus_hs_c0-199.feather
CONST_LOOKUP=$SCENE/neighbor_thirdplus_const_c40-139.feather
TRAIN_IN=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
TRAIN_OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_thirdplus_g0.0_train_full.feather
BASE_IN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather
BASE_OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v23_thirdplus_c40-199/base_c40-199.feather
mkdir -p "$SCENE" "$(dirname "$BASE_OUT")"
for f in "$HIGH" "$CONST" "$TRAIN_IN" "$BASE_IN"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

echo "### V2.3 THIRD-PLUS FEATURE PREP job=$SLURM_JOB_ID ###"; date
if [ ! -e "$LOW" ]; then
  "$PY" -u scripts/build_neighbor_flux_shells.py \
    --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
    --sign 0.0 --cases $(seq 0 39) --output "$LOW"
fi
"$PY" -u scripts/compact_neighbor_thirdplus.py \
  --inputs "$LOW" "$HIGH" --output "$HS_LOOKUP"
"$PY" -u scripts/compact_neighbor_thirdplus.py \
  --inputs "$CONST" --output "$CONST_LOOKUP"
"$PY" -u scripts/augment_catalogue_lookup.py \
  --catalogue "$TRAIN_IN" --lookup "$HS_LOOKUP" --columns nbr_flux_thirdplus \
  --output "$TRAIN_OUT"
"$PY" -u scripts/augment_catalogue_lookup.py \
  --catalogue "$BASE_IN" --lookup "$HS_LOOKUP" --columns nbr_flux_thirdplus \
  --output "$BASE_OUT"
echo V23_THIRDPLUS_PREP_DONE; date

#!/bin/bash
#SBATCH --job-name=resp_rebuild_c2fix
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_rebuild_c2fix_%j.out
set -eo pipefail
# c2-CENTROID-FIX RESP-target rebuild (WORKLOG cont.41). Rebuilds the response-penalty target on
# the FIXED (fixed-centroid) g0.05 shapes so the flow penalty stops pinning R_flow ~2.6% high
# (interim fixed-cats m = -2.6%). Three sequential products, all on fixed shapes:
#   1. g0_lookup_c0-99  (SNC g=0 shape lookup) rebuilt from the RE-MEASURED g0.0 SECONDARIES.
#      MANDATORY: compute_response_target_blend subtracts e_snc(0) PER GALAXY from e(0.05); mixing an
#      old-convention SNC with the new g0.05 val would leave the centroid offset (~0.0037 in g2)
#      uncancelled and corrupt the target by ~-0.074. Both legs must be same-convention.
#   2. det_meas_crowd_g0.05_val_full  augmented from the fixed g0.05 val (crowd_flux/blend lookups
#      are flux/geometry -> fix-invariant, reused as-is).
#   3. response_target_crowd_rblend_snc_c0-99_6x3x5.npz  (OVERWRITES the stale target).
# Old products archived to .oldcats_bak / results/*.stale_bak for comparison.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
date

echo "### [1/3] rebuild SNC g0_lookup_c0-99 from FIXED g0.0 secondaries ###"
[ -f results/g0_lookup_c0-99.feather ] && cp -a results/g0_lookup_c0-99.feather results/g0_lookup_c0-99.feather.stale_bak
python -u scripts/build_g0_lookup.py --cases $(seq 0 99) \
  --output results/g0_lookup_c0-99.feather 2>&1 | grep -v "module command"

echo "### [2/3] augment FIXED g0.05 val -> det_meas_crowd_g0.05_val_full ###"
[ -f $D/det_meas_crowd_g0.05_val_full.feather ] && mv $D/det_meas_crowd_g0.05_val_full.feather $D/det_meas_crowd_g0.05_val_full.feather.oldcats_bak
python -u scripts/augment_crowding.py --catalogue $D/det_meas_ngmix_np7_g0.05_val.feather \
  --flux-lookup results/crowd_flux_c0-199.feather \
  --blend-lookup results/blend_lookup_c0-199.feather \
  --output $D/det_meas_crowd_g0.05_val_full.feather 2>&1 | grep -v "module command"

echo "### [3/3] build RESP npz (SNC, 6x3x5, cases<=99) on FIXED shapes ###"
OUT=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
[ -f $OUT ] && cp -a $OUT ${OUT}.stale_bak
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-size 3 --n-crowd 5 --min-count 500 --max-case 99 \
  --snc-lookup results/g0_lookup_c0-99.feather \
  --output $OUT 2>&1 | grep -v "module command"
date
echo "RESP_REBUILD_C2FIX_DONE $OUT"
python -c "import numpy as np; d=np.load('$OUT'); print('global_R=', float(d['global_R']) if 'global_R' in d else 'n/a', ' keys=', list(d.keys()))" 2>/dev/null || true

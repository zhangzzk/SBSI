#!/bin/bash
#SBATCH --job-name=AP_g02
#SBATCH --time=10:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_g02_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_g02_%j.err

# STAGE 2/4 of the g=0.2 retrain (WORKLOG 2026-08-03b). Depends on job 15485354 (the sheared-half
# shape measurement) via --dependency=afterok, so it cannot start on an incomplete Shapes/ tree.
#
# Same recipe as the archived g=0.05 build (jobs/archive/job_build_allpairs_full.sh): 7" aperture,
# k = 20, --flow-only, --include-shapes, cases 0-199. Only --shear changes.
#
# NAMING: `_val` mirrors the g0.05 leg because it plays the same role (the SHEARED leg that the pair
# builder reads). Provenance note: g=0.2 comes from the `response` sim set, not `self_response`, but
# the scenes are identical -- `gals_info` is the same 699,568 galaxies with identical true properties
# at g0.0 / g0.05 / g0.2 (verified 2026-08-03).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues

echo -n "PRECHECK g0.2 secondary shape catalogues (need 200): "
N=$(find /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/case*_0.2 -path '*Shapes/shape_catalogue_detect_position_secondaries_tile*.feather' 2>/dev/null | grep -v bak | wc -l)
echo "$N"
[ "$N" -eq 200 ] || { echo "REFUSING: expected 200, found $N. Stage 1 did not finish cleanly."; exit 1; }

echo "=== AP7 build shear=0.2 (7 arcsec, k=20, flow-only, cases 0-199) ==="; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear 0.2 --cases 0-199 --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 20 \
  --n-jobs 6 --batch-size 6 \
  --output "$OUT/det_meas_ngmix_ap7_g0.2_val.feather" || exit 1
echo "=== DONE ==="; date
ls -la "$OUT/det_meas_ngmix_ap7_g0.2_val.feather"

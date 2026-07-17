#!/bin/bash
#SBATCH --job-name=AP_pilot
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_pilot_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_pilot_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "=== ALL-PAIRS PILOT build (7 arcsec, k=20, flow-only) ==="; date
for SH in 0.0 0.05; do
  echo "--- shear $SH ---"
  python -u scripts/build_detection_measurement_catalogue.py \
    --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
    --shear "$SH" --cases 0-7 --include-shapes --flow-only \
    --r-max 7 --r-min 0 --k 20 \
    --n-jobs 16 --batch-size 4 \
    --output "$OUT/det_meas_ngmix_ap7_g${SH}_pilot.feather"
done
echo "=== PILOT DONE ==="; date
ls -la "$OUT"/det_meas_ngmix_ap7_*_pilot.feather 2>/dev/null

#!/bin/bash
#SBATCH --job-name=AP_full
#SBATCH --time=08:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_full_%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_full_%x_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
SH="$1"; TAG="$2"   # e.g. 0.0 train  |  0.05 val
echo "=== ALL-PAIRS FULL build shear=$SH tag=$TAG (7 arcsec, k=20, flow-only, cases 0-199) ==="; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear "$SH" --cases 0-199 --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 20 \
  --n-jobs 6 --batch-size 6 \
  --output "$OUT/det_meas_ngmix_ap7_g${SH}_${TAG}.feather"
echo "=== DONE shear=$SH ==="; date
ls -la "$OUT/det_meas_ngmix_ap7_g${SH}_${TAG}.feather"

#!/bin/bash
#SBATCH --job-name=AP_g002
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_g002_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_g002_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "=== ALL-PAIRS g0.02 held-out test build (7 arcsec, k=20, flow-only, cases 0-19) ==="; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear 0.02 --cases 0-19 --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 20 --n-jobs 6 --batch-size 6 \
  --output "$OUT/det_meas_ngmix_ap7_g0.02_test.feather"
echo "=== DONE ==="; date; ls -la "$OUT/det_meas_ngmix_ap7_g0.02_test.feather"

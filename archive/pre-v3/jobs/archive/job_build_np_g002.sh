#!/bin/bash
#SBATCH --job-name=NP_g002
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np_g002_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "=== nearest-pair ngmix g0.02 held-out (3in/k=2, cases 0-19) ==="; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear 0.02 --cases 0-19 --include-shapes \
  --r-max 3 --r-min 0 --k 2 --n-jobs 16 --batch-size 10 \
  --output "$OUT/det_meas_ngmix_g0.02_test.feather"
echo DONE; date

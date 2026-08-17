#!/bin/bash
#SBATCH --job-name=NP7_build
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7_build_%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/np7_build_%x_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
SH="$1"; TAG="$2"; CASES="$3"
echo "=== NEAREST-PAIR 7in build shear=$SH tag=$TAG cases=$CASES (k=2, flow-only) ==="; date
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear "$SH" --cases "$CASES" --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 2 --n-jobs 8 --batch-size 10 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g${SH}_${TAG}.feather
echo DONE; date

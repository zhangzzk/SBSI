#!/bin/bash
#SBATCH --job-name=AP_g002_100
#SBATCH --time=03:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_g002_100_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_g002_100_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear 0.02 --cases 0-99 --include-shapes --flow-only \
  --r-max 7 --r-min 0 --k 20 --n-jobs 6 --batch-size 6 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_ap7_g0.02_test100.feather
echo DONE; date

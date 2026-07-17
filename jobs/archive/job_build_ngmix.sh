#!/bin/bash
#SBATCH --job-name SBSI_NGB
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_ngb_%x.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_ngb_%x.%j.err
echo "START ngmix re-aggregation shear=$SHEAR -> $OUT"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/build_detection_measurement_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --shear "$SHEAR" --cases 0-199 --include-shapes \
  --n-jobs 16 --batch-size 10 \
  --output "$OUT"
echo; echo FINISH; date

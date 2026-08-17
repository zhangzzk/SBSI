#!/bin/bash
#SBATCH --job-name=SBSI_G002_TEST
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=300G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_g002_test_%j.out
#SBATCH --partition=cluster
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_g002_test_%j.err

echo "START - g=0.02 HELD-OUT first-moment m test"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather

# Calibrate responsivity at g=0.05 (R_sim=0.2313+/-0.0007, 14M rows), predict g=0.02.
# Paired galaxies (seed=i+123 shear-independent) -> low-variance held-out test in the
# Stage-IV range. Sub-percent here = legitimate validated sub-percent calibration.
python -u SBSI/scripts/quick_rsim.py --catalogue "$CAT" --nominal-g 0.02 \
    --calib-R 0.2313 --calib-R-sem 0.0007

echo "FINISH"; date

#!/bin/bash
#SBATCH --job-name SBSI_DETMEAS
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_detmeas_%j.out
#SBATCH --partition=cluster
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_detmeas_%j.err

echo "START - SBSI detection+measured catalogue build"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Re-derive a g-specific detection+measured (k=2 nearest-pair) catalogue from the
# existing renderings of the live blendemu run (lsst_sims_fs2_25876).
SHEAR="${SHEAR:?set SHEAR=0.0|0.05|0.2}"
CASES="${CASES:-0-199}"
OUTDIR="${OUTDIR:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues}"
OUTPUT="${OUTPUT:-${OUTDIR}/det_meas_g${SHEAR}.feather}"
mkdir -p "${OUTDIR}"

python -u SBSI/scripts/build_detection_measurement_catalogue.py \
    --data-path /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
    --shear "${SHEAR}" \
    --cases "${CASES}" \
    --output "${OUTPUT}" \
    --r-max 3 --r-min 0 --k 2 \
    --n-jobs 16 \
    --batch-size 10

echo "FINISH"
date

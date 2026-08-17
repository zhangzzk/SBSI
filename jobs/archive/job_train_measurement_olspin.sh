#!/bin/bash
#SBATCH --job-name SBSI_MEAS_OLSPIN
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_meas_olspin.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_meas_olspin.%j.err

echo "START - SBSI g=0 shape-only measurement flow, OLS-frozen mean head"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather}"
OUTPUT="${OUTPUT:-SBSI/models/measurement_flow_g0_shape2d_olspin_v1.pt}"

python -u SBSI/scripts/train_measurement_model.py \
    --catalogue "$CATALOGUE" \
    --output "$OUTPUT" \
    --target-column detected \
    --selection-name sextractor_detected \
    --feature-set g0_oriented \
    --target-features measured_e1_image measured_e2_image \
    --flow-type mean_affine \
    --mean-hidden 0 \
    --freeze-mean-ols \
    --flow-blind-features e1_input_p e2_input_p \
    --shear-case 0.0 \
    --max-rows "${MAXROWS:-6000000}" \
    --epochs "${EPOCHS:-120}" \
    --batch-size "${BATCH:-8192}" \
    --hidden-dim "${HIDDEN:-256}" \
    --condition-layers "${CLAYERS:-3}" \
    --n-flows "${NFLOWS:-10}" \
    --lr "${LR:-0.0007}" \
    --patience "${PATIENCE:-15}" \
    --num-workers 8

echo "FINISH"
date

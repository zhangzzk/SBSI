#!/bin/bash
#SBATCH --job-name SBSI_MEAS_G0
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_measurement_g0.%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_measurement_g0.%j.err

echo "START - SBSI g=0 shear-free measurement flow"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather}"
FEATURESET="${FEATURESET:-g0_oriented}"
OUTPUT="${OUTPUT:-SBSI/models/measurement_flow_${FEATURESET}_v1.pt}"

python -u SBSI/scripts/train_measurement_model.py \
    --catalogue "$CATALOGUE" \
    --output "$OUTPUT" \
    --target-column detected \
    --selection-name sextractor_detected \
    --feature-set "$FEATURESET" \
    ${TARGETS:+--target-features $TARGETS} \
    --flow-type "${FLOWTYPE:-affine}" \
    --num-bins "${NUMBINS:-8}" \
    --mean-hidden "${MEANHIDDEN:-0}" \
    ${BLINDFEATS:+--flow-blind-features $BLINDFEATS} \
    --shear-case 0.0 \
    --max-rows "${MAXROWS:-4000000}" \
    --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH:-8192}" \
    --hidden-dim "${HIDDEN:-256}" \
    --condition-layers "${CLAYERS:-3}" \
    --n-flows "${NFLOWS:-10}" \
    --lr "${LR:-0.0007}" \
    --patience "${PATIENCE:-12}" \
    --num-workers 8

echo "FINISH"
date

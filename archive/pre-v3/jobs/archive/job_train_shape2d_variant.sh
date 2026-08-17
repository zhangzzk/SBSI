#!/bin/bash
#SBATCH --job-name SBSI_M_VAR
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_m_var.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_m_var.%j.err

echo "START - SBSI shape-only measurement flow variant"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather}"
OUTPUT="${OUTPUT:?set OUTPUT}"
# shape-only targets; residual flow BLIND to the shape inputs by default (mean head carries
# the shape->shape response). BLIND="none" to let the residual flow see shape.
# Accept comma-separated (avoids spaces in --export); converted to space-separated here.
BLIND="${BLIND:-e1_input_p,e2_input_p}"
[ "$BLIND" = "none" ] && BLIND=""
BLIND="${BLIND//,/ }"

python -u SBSI/scripts/train_measurement_model.py \
    --catalogue "$CATALOGUE" \
    --output "$OUTPUT" \
    --target-column detected --selection-name sextractor_detected \
    --feature-set g0_oriented \
    --target-features ${TARGETS:-measured_e1_image measured_e2_image} \
    --flow-type "${FLOWTYPE:-mean_affine}" \
    --mean-hidden "${MEANHIDDEN:-0}" \
    ${BLIND:+--flow-blind-features $BLIND} \
    ${FREEZE:+--freeze-mean-ols} \
    ${DECORR:+--decorrelate-shape-size} \
    --shear-case 0.0 \
    --max-rows "${MAXROWS:-6000000}" \
    --epochs "${EPOCHS:-150}" \
    --batch-size "${BATCH:-8192}" \
    --hidden-dim "${HIDDEN:-256}" \
    --condition-layers "${CLAYERS:-3}" \
    --n-flows "${NFLOWS:-10}" \
    --lr "${LR:-0.0007}" \
    --patience "${PATIENCE:-15}" \
    --num-workers 8

echo "FINISH"; date

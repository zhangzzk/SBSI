#!/bin/bash
#SBATCH --job-name SBSI_M_RESP
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_m_resp.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_m_resp.%j.err

echo "START - SBSI response-aware measurement flow (NLL + lambda*||R_model - R_sim||^2)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CATALOGUE="${CATALOGUE:-/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather}"
OUTPUT="${OUTPUT:?set OUTPUT}"
BLIND="${BLIND:-e1_input_p,e2_input_p}"
[ "$BLIND" = "none" ] && BLIND=""
BLIND="${BLIND//,/ }"

# Response-aware (Sobolev): same architecture as the meanblind baseline, plus the response
# loss pulling the model's induced first-moment response R_model -> R_sim(0.05)=0.2313.
python -u SBSI/scripts/train_measurement_model.py \
    --catalogue "$CATALOGUE" \
    --output "$OUTPUT" \
    --target-column detected --selection-name sextractor_detected \
    --feature-set g0_oriented \
    --target-features ${TARGETS:-measured_e1_image measured_e2_image} \
    --flow-type "${FLOWTYPE:-mean_affine}" \
    --mean-hidden "${MEANHIDDEN:-0}" \
    ${BLIND:+--flow-blind-features $BLIND} \
    --shear-case 0.0 \
    --max-rows "${MAXROWS:-6000000}" \
    --epochs "${EPOCHS:-150}" \
    --batch-size "${BATCH:-8192}" \
    --hidden-dim "${HIDDEN:-256}" \
    --condition-layers "${CLAYERS:-3}" \
    --n-flows "${NFLOWS:-10}" \
    --lr "${LR:-0.0007}" \
    --patience "${PATIENCE:-20}" \
    --num-workers 8 \
    --response-weight "${LAMBDA:-50}" \
    --response-target "${RTARGET:-0.2313}" \
    --response-delta "${RDELTA:-0.05}" \
    ${RNPZ:+--response-target-npz $RNPZ}

echo "FINISH"; date

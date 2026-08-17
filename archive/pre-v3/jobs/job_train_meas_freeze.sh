#!/bin/bash
#SBATCH --job-name=train_frz
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_frz_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# FROZEN-MEAN variant (WORKLOG cont.26): the ~1% per-seed scatter in m lives in the SGD-trained
# mean head (= R_flow). --freeze-mean-ols fits the linear mean head by OLS on the fixed g=0 data
# and FREEZES it, making R_flow DETERMINISTIC (data-fixed, seed-independent). It is an ALTERNATIVE
# to L_response (line 664: cannot combine with --response-weight) -- a different fix for the same
# measured-shape response under-fit. Same feature set / flow / blind-features as the round-2 recipe,
# but NO response-weight/target and no --response-* args. Two seeds test that R_flow is reproducible.
FS=${FS:?set FS}
SEED=${SEED:-421}
SUF=${SEED_SUFFIX:-}
TAG=${TAG:?set TAG}_frzols${SUF}
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_v1.pt

echo "### TRAIN FREEZE $TAG  feature-set=$FS  seed=$SEED (freeze-mean-ols, no response loss) ###"
echo "catalogue=$CAT"; date
python -u scripts/train_measurement_model.py \
  --catalogue $CAT \
  --output $OUT \
  --target-column detected \
  --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine \
  --mean-hidden 0 \
  --flow-blind-features e1_input_p e2_input_p \
  --freeze-mean-ols \
  --shear-case 0.0 \
  --max-rows 0 \
  --epochs 80 \
  --batch-size 8192 \
  --hidden-dim 256 \
  --condition-layers 3 \
  --n-flows 10 \
  --lr 0.0007 \
  --patience 10 \
  --seed "$SEED" \
  --num-workers 8 || { echo "TRAIN_FRZ FAILED TAG=$TAG"; exit 1; }
date
echo "TRAIN_FRZ_DONE $OUT"

#!/bin/bash
#SBATCH --job-name=train_relerr
#SBATCH --time=30:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_relerr_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# FLATNESS retrain: conc feature (nbr_flux_max) + RELATIVE response error. The absolute-loss conc
# run hit global +0.11% but by rebalancing (negative low-blend cancels residual q3 +4.1%), spread
# ~6.7%. Relative error penalizes per-bin m = ((R_model-R_sim)/R_sim)^2, driving each r_blend bin
# to 0 uniformly -> should flatten instead of rebalance (docstring: robust to population reweighting).
# Only change vs job_train_crowd_conc.sh: --response-error relative; batch 16384 (2x, faster on 2080 Ti).
FS=g0_crowd_flux_conc
TAG=crowdflux_conc_snc100_central02_relerr
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN CONC-RELERR $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
echo "catalogue=$CAT"; echo "response_target=$RESP"; date
python -u scripts/train_measurement_model.py \
  --catalogue $CAT \
  --output $OUT \
  --target-column detected \
  --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine \
  --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 \
  --max-rows 0 \
  --epochs "$EPOCHS" \
  --batch-size 16384 \
  --hidden-dim 256 \
  --condition-layers 3 \
  --n-flows 10 \
  --lr 0.0007 \
  --patience 10 \
  --num-workers 8 \
  --response-weight "$LAM" \
  --response-delta "$DELTA" \
  --response-difference central \
  --response-error relative \
  --response-target-npz "$RESP" || { echo "TRAIN_RELERR FAILED"; exit 1; }
date
echo "TRAIN_RELERR_DONE $OUT"

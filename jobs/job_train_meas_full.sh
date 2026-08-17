#!/bin/bash
#SBATCH --job-name=train_full
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_full_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# REALISTIC conditioning V2 / TARGET (WORKLOG cont.19). IDENTICAL to job_train_crowd_conc.sh (conc-v1)
# EXCEPT --feature-set and --output. Fully realistic primary own-props: measured_mag_auto +
# measured_flux_radius + measured_class_star; DROP true sersic_n + redshift. Intrinsic e1/e2_input_p
# KEPT flow-blind (mean-head-only shear channel). Neighbours (theta_blending) TRUE. This is the
# p(ehat,thetahat|e,theta,theta_blending) go/no-go for real-data realism.
FS=g0_meas_crowd_conc_full
TAG=meas_full_snc100_central02
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN MEAS-FULL $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
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
  --batch-size 8192 \
  --hidden-dim 256 \
  --condition-layers 3 \
  --n-flows 10 \
  --lr 0.0007 \
  --patience 10 \
  --num-workers 8 \
  --response-weight "$LAM" \
  --response-delta "$DELTA" \
  --response-difference central \
  --response-target-npz "$RESP" || { echo "TRAIN_MEAS_FULL FAILED"; exit 1; }
date
echo "TRAIN_MEAS_FULL_DONE $OUT"

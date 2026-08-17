#!/bin/bash
#SBATCH --job-name=train_g02tgt
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_g02tgt_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# STEP 2 of the g=0.02-target experiment (WORKLOG cont.17). IDENTICAL recipe to the accepted conc-v1
# (job_train_crowd_conc.sh) -- same feature set g0_crowd_flux_conc, same training catalogue, same
# lam=300, delta=0.02, central, 80 epochs, batch 8192 -- EXCEPT the response target is now the g=0.02 /
# cases-0-199 table. Batch kept at 8192 (NOT 16384) on purpose so this is a clean A/B vs conc-v1 and
# the ONLY changed variable is the target amplitude. The delta stays 0.02, which now MATCHES the
# target amplitude and the constgold validation amplitude (the point of the experiment).
FS=g0_crowd_flux_conc
TAG=crowdflux_conc_tgt02c99_central02
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz   # from job_resp_target_g02.sh
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather                  # SAME as conc-v1 (has nbr_flux_max)
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN g02-target $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
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
  --response-target-npz "$RESP" || { echo "TRAIN_G02TGT FAILED"; exit 1; }
date
echo "TRAIN_G02TGT_DONE $OUT"

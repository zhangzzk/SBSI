#!/bin/bash
#SBATCH --job-name=train_szfl
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_szfl_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# REALISTIC conditioning V1 / BRIDGE (WORKLOG cont.19). IDENTICAL to job_train_crowd_conc.sh (conc-v1)
# EXCEPT --feature-set and --output. Swaps ONLY the primary size+flux true->measured
# (r_input_p_scaled->measured_mag_auto, Re_input_p_scaled->measured_flux_radius); keeps sersic_n +
# redshift TRUE. Intrinsic e1/e2_input_p KEPT flow-blind (mean-head-only shear channel). Neighbours TRUE.
FS=g0_meas_crowd_conc_szfl
TAG=meas_szfl_snc100_central02
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN MEAS-SZFL $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
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
  --response-target-npz "$RESP" || { echo "TRAIN_MEAS_SZFL FAILED"; exit 1; }
date
echo "TRAIN_MEAS_SZFL_DONE $OUT"

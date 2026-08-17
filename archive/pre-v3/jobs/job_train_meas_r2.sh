#!/bin/bash
#SBATCH --job-name=train_r2
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/train_r2_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# ROUND 2 realistic-conditioning retrains (WORKLOG cont.20). Parametrized clone of job_train_meas_full.sh;
# FS + TAG passed via sbatch --export=ALL,FS=...,TAG=...  (v3a=g0_meas_conc_sern/meas_sern,
# v3b=g0_meas_conc_z/meas_z, v4=g0_meas_conc_struct/meas_conc). Everything else IDENTICAL to conc-v1.
FS=${FS:?set FS}
SEED=${SEED:-421}                       # noise-floor study passes SEED=422/423/... to measure retrain scatter
SUF=${SEED_SUFFIX:-}                     # e.g. _s422 so seed repeats don't overwrite each other; empty = default
TAG=${TAG:?set TAG}_snc100_central02${SUF}
# Realistic-structure study (WORKLOG cont.24): inject survey measurement noise on the TRUE
# structure conditioners + optionally train on a case subset for a fast first signal.
NOISE_PHOTOZ=${NOISE_PHOTOZ:-0}          # sigma_z = this*(1+z) on redshift_input_p
NOISE_SERSIC=${NOISE_SERSIC:-0}          # fractional sigma = this*|n| on sersic_n_input_p
MAX_CASES=${MAX_CASES:-}                 # keep only case < this (empty = all 200)
XARGS=""
[ "$NOISE_PHOTOZ" != "0" ] && XARGS="$XARGS --noise-photoz $NOISE_PHOTOZ"
[ "$NOISE_SERSIC" != "0" ] && XARGS="$XARGS --noise-sersic-frac $NOISE_SERSIC"
[ -n "$MAX_CASES" ]        && XARGS="$XARGS --max-cases $MAX_CASES"
LAM=300
DELTA=0.02
EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### TRAIN R2 $TAG  feature-set=$FS  lam=$LAM  delta=$DELTA epochs=$EPOCHS ###"
echo "noise: photoz=$NOISE_PHOTOZ sersic_frac=$NOISE_SERSIC  max_cases=${MAX_CASES:-all}  xargs='$XARGS'"
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
  --seed "$SEED" \
  --num-workers 8 \
  --response-weight "$LAM" \
  --response-delta "$DELTA" \
  --response-difference central \
  $XARGS \
  --response-target-npz "$RESP" || { echo "TRAIN_R2 FAILED TAG=$TAG"; exit 1; }
date
echo "TRAIN_R2_DONE $OUT"

#!/bin/bash
#SBATCH --job-name=ens_seed
#SBATCH --time=12:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/ens_seed_%j.out

# SEED-ENSEMBLE certification (WORKLOG cont.28). The per-seed m scatter (+/-1%) is training
# stochasticity (SGD/cuDNN over 80 epochs), NOT feature choice -- confirmed universal in cont.26.
# The frozen-OLS pivot (cont.26) removed the scatter but a FLAT linear response is biased
# (R_flow=0.3289 -> m=-6.56%), so it cannot reach sub-percent. The honest deterministic route is
# to ENSEMBLE the L_response MLP seeds: mean R_flow has error sigma/sqrt(N). N~12 -> +/-0.3%.
# This ONE job trains V3a (g0_meas_conc_sern = measured primaries + TRUE sersic + TRUE neighbours,
# the realistic recipe) for a single SEED, then validates on the clean 40-139 split and prints R_flow.
# Harvest R_flow from all seed logs and average offline -> certified deterministic m.
# NOTE: do NOT use `set -u` -- conda's activate.d scripts reference unbound vars (ADDR2LINE) and abort.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

SEED=${SEED:?set SEED}
FS=${FS:-g0_meas_conc_sern}
TAG=${TAG:-meas_sern}_ens_s${SEED}
LAM=300; DELTA=0.02; EPOCHS=80
RESP=results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CAT=$D/det_meas_crowd_conc_g0.0_train_full.feather
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
OUT=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt
MP=results/meas_prim_lookup_c0-139.feather
FL=results/crowd_flux_conc_c0-199.feather

echo "### ENSEMBLE SEED $SEED  FS=$FS  TAG=$TAG ###"; date
python -u scripts/train_measurement_model.py \
  --catalogue $CAT --output $OUT \
  --target-column detected --selection-name sextractor_detected \
  --feature-set "$FS" \
  --target-features measured_ngmix_g1 measured_ngmix_g2 \
  --flow-type mean_affine --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 0 --epochs "$EPOCHS" --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 10 \
  --seed "$SEED" --num-workers 8 \
  --response-weight "$LAM" --response-delta "$DELTA" --response-difference central \
  --response-target-npz "$RESP" || { echo "ENS_SEED TRAIN FAILED seed=$SEED"; exit 1; }
echo "TRAIN_DONE seed=$SEED"; date

echo "### VALIDATE seed=$SEED on clean 40-139 ###"
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $OUT \
  --catalogue $CD/constant_response_catalogue_train.feather --min-case 40 \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup $FL \
  --meas-prim-lookup $MP \
  --ood-lookup results/ood_split_c40-139.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --global-only \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo "ENS_SEED_DONE seed=$SEED MODEL=$OUT"

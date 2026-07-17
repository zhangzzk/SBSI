#!/bin/bash
#SBATCH --job-name=val_s1
#SBATCH --time=10:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_s1_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# NOISE-FLOOR study (WORKLOG cont.21): validate ONE model on the CLEAN emulator-in-sample split (40-139)
# ONLY -- STEP 2 (held-out) dropped to halve wall time, since seed scatter (not emulator generalization)
# is what we are measuring. MODEL passed via --export; MP (meas-prim lookup) empty for TRUE-prop feature
# sets (conc-v1) and set for measured variants (V2). FL = crowd-flux lookup (TRUE neighbours).
MODEL=${MODEL:?set MODEL}
FL=${FL:-results/crowd_flux_conc_c0-199.feather}
MP=${MP:-}
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
MPARG=""; [ -n "$MP" ] && MPARG="--meas-prim-lookup $MP"
# Realistic-structure study: MUST match the noise the model was TRAINED with (WORKLOG cont.24).
NOISE_PHOTOZ=${NOISE_PHOTOZ:-0}; NOISE_SERSIC=${NOISE_SERSIC:-0}
NARG=""
[ "$NOISE_PHOTOZ" != "0" ] && NARG="$NARG --noise-photoz $NOISE_PHOTOZ"
[ "$NOISE_SERSIC" != "0" ] && NARG="$NARG --noise-sersic-frac $NOISE_SERSIC"
echo "### VAL STEP1(40-139) MODEL=$MODEL  MP='$MP'  noise(photoz=$NOISE_PHOTOZ,sersic=$NOISE_SERSIC) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_c40-139.feather \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup $FL \
  $MPARG \
  $NARG \
  --ood-lookup results/ood_split_c40-139.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo "VAL_STEP1_DONE MODEL=$MODEL"

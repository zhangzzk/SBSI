#!/bin/bash
#SBATCH --job-name=val_g02tgt
#SBATCH --time=18:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_g02tgt_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# STEP 3 of the g=0.02-target experiment (WORKLOG cont.17). Validate the g02-target flow on constgold,
# NO deficit(R_blend) correction. IDENTICAL to job_validate_conc.sh except MODEL. Direct A/B vs conc-v1:
#   conc-v1 (g=0.05 target): 100-case(40-139) +0.11% ; held-out(0-39) -0.07% ; per-bin spread ~6.7/10.7%.
# Success = same-or-better GLOBAL with a SMALLER per-bin spread (amplitude fix should flatten, not just
# rebalance) -- watch the ISO/q1..q4 quintile row, not only global m.
MODEL=models/measurement_flow_g0_ngmix_crowdflux_conc_tgt02c99_central02_lam300_v1.pt
FL=results/crowd_flux_conc_c0-199.feather          # near/far/MAX (must match training feature set)
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant

echo "### STEP 1: 100-case (40-139) with g02-target flow, NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_c40-139.feather \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup $FL \
  --ood-lookup results/ood_split_c40-139.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date

echo "### STEP 2: emulator-HELD-OUT (0-39) with g02-target flow, NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_train.rmax10.feather \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup $FL \
  --ood-lookup results/ood_split_c0-39.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo VAL_G02TGT_DONE

#!/bin/bash
#SBATCH --job-name=val_conc
#SBATCH --time=18:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_conc_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# Validate the CONCENTRATION-retrained flow on constgold, NO deficit(R_blend) correction.
# Root-cause test: does adding nbr_flux_max flatten q3 at the source?
#   Baselines (old flow, uncorrected):  100-case(40-139) +0.81%, q3 +6.8% ; held-out(0-39) +1.63%.
#   Bolt-on correction reached:          100-case +0.28% ; held-out +1.16% (q3 still +3.7%).
MODEL=models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt
FL=results/crowd_flux_conc_c0-199.feather          # near/far/MAX (must match training feature set)
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant

echo "### STEP 1: 100-case (40-139) with conc flow, NO correction ###"; date
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

echo "### STEP 2: emulator-HELD-OUT (0-39) with conc flow, NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_train.rmax10.feather \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup $FL \
  --ood-lookup results/ood_split_c0-39.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo VAL_CONC_DONE

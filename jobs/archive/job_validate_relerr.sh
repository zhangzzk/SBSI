#!/bin/bash
#SBATCH --job-name=val_relerr
#SBATCH --time=18:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/val_relerr_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI

# Validate the RELATIVE-error conc flow on constgold, NO correction. Goal = FLAT per-bin profile
# (small spread), not just small global. Compare vs conc-absolute (global +0.11%, spread ~6.7%, q3 +4.1%).
MODEL=models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_relerr_lam300_v1.pt
FL=results/crowd_flux_conc_c0-199.feather
CD=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant

echo "### STEP 1: 100-case (40-139) relerr conc flow, NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_c40-139.feather \
  --blend-lookup results/blend_lookup_extnbrho_c40-139.feather \
  --crowd-flux-lookup $FL \
  --ood-lookup results/ood_split_c40-139.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 45000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date

echo "### STEP 2: emulator-HELD-OUT (0-39) relerr conc flow, NO correction ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL \
  --catalogue $CD/constant_response_catalogue_train.rmax10.feather \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup $FL \
  --ood-lookup results/ood_split_c0-39.feather \
  --n-samples 64 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 300 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo VAL_RELERR_DONE

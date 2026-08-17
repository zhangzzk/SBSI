#!/bin/bash
#SBATCH --job-name=qdiag_orth
#SBATCH --time=10:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/qdiag_orth_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# CONFIRM-FIRST diagnostic for the flow-retrain decision.
# Question: at FIXED nbr_flux (the flow's only crowding input), does the flow deficit
# (R_sim - R_flow - R_blend) still rise with n_pairs? If yes -> n_pairs carries info the flow
# cannot see -> adding it as a conditioning feature is justified before committing to a retrain.
# Run on cases 40-79 (in-emulator-training, cleanest & largest q3 deficit signal).
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_c40-79.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL --catalogue $CAT \
  --blend-lookup results/blend_lookup_extnbrho_c40-79.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather \
  --ood-lookup results/ood_split_c40-79.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --n-samples 32 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 200 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo QDIAG_ORTH_DONE

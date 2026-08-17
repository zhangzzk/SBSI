#!/bin/bash
#SBATCH --job-name=ortho_conc
#SBATCH --time=06:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/ortho_conc_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# CLEAN mechanistic check: re-run the orthogonality diagnostic for the CONC flow on cases 40-79
# ONLY (matches the multiplicity lookup coverage, so n_pairs is real for every row -- unlike the
# 40-139 validation where cases 80-139 have unmatched n_pairs=0 and confound the split).
# Baseline (old flow, job 15029001) Δdeficit(nhi-nlo) = -0.0071, -0.0071, -0.0124 across flux Q1-Q3.
# If the conc feature worked, these shrink toward 0 (the flow now sees concentration).
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_c40-79.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model $MODEL --catalogue $CAT \
  --blend-lookup results/blend_lookup_extnbrho_c40-79.feather \
  --crowd-flux-lookup results/crowd_flux_conc_c0-199.feather \
  --ood-lookup results/ood_split_c40-79.feather \
  --mult-lookup results/blend_multiplicity_extnbrho_c40-79.feather \
  --n-samples 32 --n-blend 4 --blend-eps 0.02 --max-rows 40000000 --n-boot 200 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo ORTHO_CONC_DONE

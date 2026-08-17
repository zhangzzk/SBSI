#!/bin/bash
#SBATCH --job-name=qdiag_def
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/qdiag_def_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_c40-79.feather
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
NPZ=results/deficit_rblend_c40-59.npz
COMMON="--measurement-model $MODEL --catalogue $CAT \
  --blend-lookup results/blend_lookup_extnbrho_c40-79.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c40-79.feather \
  --n-samples 48 --n-blend 4 --blend-eps 0.02 --max-rows 40000000"

echo "### STEP 1: FIT deficit(R_blend) on cases 40-59 (24 bins) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON \
  --min-case 40 --max-case 60 --fit-rblend-corr "$NPZ" --corr-nbin 24 2>&1 \
  | grep --line-buffered -vE "module command"
date

echo "### STEP 2: APPLY on held-out cases 60-79 (out-of-sample) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON \
  --min-case 60 --max-case 80 --apply-rblend-corr "$NPZ" --n-boot 200 2>&1 \
  | grep --line-buffered -vE "module command"
date

echo "### STEP 3: CONTROL - held-out 60-79 with NO correction (baseline) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py $COMMON \
  --min-case 60 --max-case 80 --n-boot 200 2>&1 \
  | grep --line-buffered -vE "module command"
date; echo QDIAG_DEF_DONE

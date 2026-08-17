#!/bin/bash
#SBATCH --job-name=fithalf
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/fithalf_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py --measurement-model models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt --blend-lookup results/blend_lookup_extnbrho_c0-39.feather --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather --n-samples 64 --n-blend 4 --blend-eps 0.02   --max-case 20 --fit-rblend-corr results/rblend_corr_c0-19.npz --corr-nbin 24 --max-rows 12000000 --n-boot 40
date; echo FITHALF_DONE

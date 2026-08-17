#!/bin/bash
#SBATCH --job-name=applyhalf
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/applyhalf_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
date; echo "### APPLY corr (fit on 0-19) to HELD-OUT cases 20-39 ###"
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py --measurement-model models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt --blend-lookup results/blend_lookup_extnbrho_c0-39.feather --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather --n-samples 64 --n-blend 4 --blend-eps 0.02   --min-case 20 --apply-rblend-corr results/rblend_corr_c0-19.npz --max-rows 12000000 --n-boot 200 2>&1 | grep -vE "module command"
date; echo APPLYHALF_DONE

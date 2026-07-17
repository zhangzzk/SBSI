#!/bin/bash
#SBATCH --job-name=cg_fit
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_fit_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
echo "### CONSTGOLD FIT deficit(R_blend) on cases 0-39 (fit-only) ###"; date
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather \
  --fit-rblend-corr results/rblend_corr_c0-39.npz --corr-nbin 24 \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 50
date; echo "CG_FIT_DONE"

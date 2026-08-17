#!/bin/bash
#SBATCH --job-name=cg_mult
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_mult_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
echo "### CONSTGOLD MULTIPLICITY DISCRIMINATOR  model=$MODEL ###"; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather \
  --rblend-edges-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz \
  --mult-lookup results/blend_multiplicity_extnbrho_c0-39.feather \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v "module command"
date; echo "CG_MULT_DONE"

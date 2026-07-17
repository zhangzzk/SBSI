#!/bin/bash
#SBATCH --job-name=cg_mfast
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_mfast_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_lam300_v1.pt
echo "### CONSTGOLD MULT FAST (3M rows, unbuffered, branch answer) ###"; date
# NOTE: no grep pipe -> output flushes live so tables are readable as they print.
stdbuf -oL -eL python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather \
  --rblend-edges-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz \
  --mult-lookup results/blend_multiplicity_extnbrho_c0-39.feather \
  --max-rows 3000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 50
date; echo "CG_MFAST_DONE"

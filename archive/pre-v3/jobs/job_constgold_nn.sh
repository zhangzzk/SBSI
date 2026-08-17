#!/bin/bash
#SBATCH --job-name=cg_nn
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_nn_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
echo "### CONSTANT-GOLD crowd_flux : TRUTH-isolation self-response residual (per-bin bootstrap) ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --nn-lookup results/nn_dist_const_c0-39.feather \
  --nn-radii 0 2 3 5 7 10 \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v module
echo CG_NN_DONE

#!/bin/bash
#SBATCH --job-name=c_origin
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/c_origin_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/c_origin_%j.err

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

echo "### ADDITIVE ORIGIN DIAGNOSTIC: gold/g0 measured ngmix vs flow zero-shear mean ###"
python -u scripts/diagnose_additive_origin.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --ood-lookup results/ood_split_c0-39.feather \
  --nn-lookup results/nn_dist_const_c0-39.feather \
  --max-rows 12000000 --model-rows 1500000 --n-samples 128 --batch-size 65536 \
  2>&1 | grep -v "module command"
echo C_ORIGIN_DONE

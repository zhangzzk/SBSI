#!/bin/bash
#SBATCH --job-name=cg_crowd
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_crowd_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
TAG=${1:?tag}
M=models/measurement_flow_g0_ngmix_${TAG}_lam300_v1.pt
echo "### CONSTANT-GOLD $TAG : crowd flow (+) blend emulator (r<28), binned by R_blend ###"
python -u scripts/validate_constant_with_blend.py --measurement-model "$M" \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --max-rows 6000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 2>&1 | grep -v module
echo "CG_CROWD_DONE $TAG"

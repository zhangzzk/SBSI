#!/bin/bash
#SBATCH --job-name=extnbr_val
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/extnbr_val_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### VALIDATE gold with EXTENDED-DOMAIN emulator blend (crowdflux flow, all cases 0-39) ###"
echo "### baseline (production emulator, blend_lookup_const28) gave m = +6.6% ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_extnbr_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --ood-lookup results/ood_split_c0-39.feather \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v "module command"
echo EXTNBR_VAL_DONE

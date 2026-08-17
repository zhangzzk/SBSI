#!/bin/bash
#SBATCH --job-name=cg_coh
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_coh_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### CONSTANT-GOLD crowdcoh (coherent-calibrated flow), HELD-OUT cases 20-39, R_total=R_flow ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdcoh_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --min-case 20 --no-blend --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v module
echo CG_COH_DONE

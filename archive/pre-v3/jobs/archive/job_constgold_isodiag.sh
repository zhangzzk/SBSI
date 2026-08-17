#!/bin/bash
#SBATCH --job-name=cg_isodiag
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_isodiag_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
echo "### CONSTANT-GOLD truly-isolated diagnostic (NP7 + r<28 blend, binned by R_blend) ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --max-rows 6000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 2>&1 | grep -v module
echo CG_ISODIAG_DONE

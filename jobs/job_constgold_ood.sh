#!/bin/bash
#SBATCH --job-name=cg_ood
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_ood_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
TAG=${1:?tag}; MAXOOD=${2:-}
EXTRA=""; LBL="FULL(all-40)"
if [ -n "$MAXOOD" ]; then EXTRA="--ood-lookup results/ood_lookup_const_c0-39.feather --max-ood $MAXOOD"; LBL="IN-DOMAIN(max_ood=$MAXOOD)"; fi
M=models/measurement_flow_g0_ngmix_${TAG}_lam300_v1.pt
echo "### CONSTANT-GOLD $TAG  $LBL  (all 40 cases, bootstrap error) ###"
python -u scripts/validate_constant_with_blend.py --measurement-model "$M" \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 $EXTRA 2>&1 | grep -v module
echo "CG_OOD_DONE ${TAG}_${LBL}"

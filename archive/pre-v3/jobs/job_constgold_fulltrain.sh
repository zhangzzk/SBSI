#!/bin/bash
#SBATCH --job-name=constgold_fulltrain
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/constgold_fulltrain_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

TAG=${1:-crowdflux_full200}
LAM=${2:-300}
MODEL=models/measurement_flow_g0_ngmix_${TAG}_lam${LAM}_v1.pt

echo "### VALIDATE constant-gold with full-trained crowdflux flow (cases 0-39 gold) ###"
echo "model=$MODEL"
date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather \
  --ood-lookup results/ood_split_c0-39.feather \
  --rblend-edges-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz \
  --max-rows 12000000 \
  --n-samples 64 \
  --n-blend 4 \
  --blend-eps 0.02 \
  --n-boot 200 2>&1 | grep -v "module command"
date
echo "CONSTGOLD_FULLTRAIN_DONE $TAG"

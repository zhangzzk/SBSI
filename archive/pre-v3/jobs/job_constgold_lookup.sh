#!/bin/bash
#SBATCH --job-name=cg_lookup
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_lookup_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
MODEL=${1:?model}; LOOKUP=${2:?lookup}; TAG=${3:-cg}
echo "### CONSTGOLD  model=$MODEL  blend-lookup=$LOOKUP  ($TAG) ###"; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$MODEL" --blend-lookup "$LOOKUP" \
  --crowd-flux-lookup results/crowd_flux_c0-199.feather --ood-lookup results/ood_split_c0-39.feather \
  --rblend-edges-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v "module command"
date; echo "CG_LOOKUP_DONE $TAG"

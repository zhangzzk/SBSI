#!/bin/bash
#SBATCH --job-name=sbsi_c_mean
#SBATCH --partition=cip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/z/Zekang.Zhang/logs/c_meanhead_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/c_meanhead_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"

cd /home/z/Zekang.Zhang/SBSI

OUT_MODEL="models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1.pt"

echo "### FINETUNE ADDITIVE MEAN HEAD"
date

python -u scripts/finetune_additive_mean_head.py \
  --input-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --output "${OUT_MODEL}" \
  --fit-max-case 19 \
  --eval-min-case 20 \
  --max-rows 12000000 \
  --train-rows 1500000 \
  --eval-rows 750000 \
  --epochs 30 \
  --patience 6 \
  --batch-size 65536 \
  --num-workers 4 \
  --residual-samples 128 \
  --mean-weight 50000 \
  --nll-weight 1.0 \
  --lr 0.0002

echo "### DIAGNOSE ADDITIVE ORIGIN WITH MEAN-HEAD-TUNED FLOW"
date

python -u scripts/diagnose_additive_origin.py \
  --measurement-model "${OUT_MODEL}" \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --ood-lookup results/ood_split_c0-39.feather \
  --nn-lookup results/nn_dist_const_c0-39.feather \
  --max-rows 12000000 \
  --model-rows 1500000 \
  --n-samples 128 \
  --batch-size 65536

echo "### C_MEANHEAD_DONE"
date

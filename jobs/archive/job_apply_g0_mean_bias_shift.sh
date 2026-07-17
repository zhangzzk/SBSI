#!/bin/bash
#SBATCH --job-name=sbsi_c_bias
#SBATCH --partition=cip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/z/Zekang.Zhang/logs/c_bias_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/c_bias_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"

cd /home/z/Zekang.Zhang/SBSI

OUT_MODEL="models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_bias_v1.pt"

echo "### APPLY G0 GLOBAL MEAN-HEAD BIAS SHIFT"
date

python -u scripts/apply_g0_mean_bias_shift.py \
  --input-model models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1.pt \
  --output "${OUT_MODEL}" \
  --fit-max-case 19 \
  --max-rows 12000000 \
  --model-rows 1500000 \
  --n-samples 128 \
  --batch-size 65536

echo "### DIAGNOSE ADDITIVE ORIGIN WITH BIAS-SHIFTED FLOW"
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

echo "### C_BIAS_DONE"
date

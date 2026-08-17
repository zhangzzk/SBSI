#!/bin/bash
#SBATCH --job-name=sbsi_c_fit
#SBATCH --partition=cip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/z/Zekang.Zhang/logs/c_fit_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/c_fit_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"

cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

echo "### FIT G0 ADDITIVE MEAN CORRECTION"
date

python -u scripts/fit_additive_correction.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --ood-lookup results/ood_split_c0-39.feather \
  --nn-lookup results/nn_dist_const_c0-39.feather \
  --output results/additive_correction_g0_crowdflux_4x3x5.npz \
  --fit-max-case 19 \
  --eval-min-case 20 \
  --max-rows 12000000 \
  --model-rows 1500000 \
  --n-samples 128 \
  --batch-size 65536 \
  2>&1 | grep -v "module command"

echo "### C_FIT_DONE"
date

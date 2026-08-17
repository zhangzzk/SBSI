#!/bin/bash
#SBATCH --job-name=sbsi_m_split
#SBATCH --partition=cip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/z/Zekang.Zhang/logs/m_split_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/m_split_%j.err

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"

cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

echo "### CASE-SPLIT R_BLEND RESIDUAL CALIBRATION"
date

python -u scripts/calibrate_blend_residual_split.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_extnbrho_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --fit-max-case 19 \
  --eval-min-case 20 \
  --max-rows 12000000 \
  --n-blend 4 \
  --blend-eps 0.02 \
  --n-samples 64 \
  --batch-size 65536 \
  --n-boot 300

echo "### M_SPLIT_DONE"
date

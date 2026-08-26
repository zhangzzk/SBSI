#!/bin/bash
#SBATCH --job-name=sbsi_rmatrix
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_rmatrix_%j.out

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate py31
cd /home/z/Zekang.Zhang/SBSI

python -u scripts/audit_simulation_component_response.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.05_val_full.feather \
  --max-case 99 \
  --primary-mag-max 25.8 \
  --primary-re-min 0.5 \
  --output doc/generated/simulation_component_response_v22.json

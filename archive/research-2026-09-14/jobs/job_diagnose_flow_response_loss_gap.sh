#!/usr/bin/env bash

#SBATCH --job-name=sbsi_EresponseGap
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" -u "$repo/scripts/diagnose_flow_response_loss_gap.py" \
  --model "$root/measurement_flow_plain_direct_r100_t10_d16_s501_swaavg.pt" \
  --g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --sheared-catalogue "$source_root/det_meas_crowd_conc_g0.05_all200.feather" \
  --response-target "$source_root/response_target_mixed_train160_6x6x5_matrix.npz" \
  --output "$root/response_loss_gap_r100_t10_d16_s501_fullval.json" \
  --device cuda --delta 0.02 --batch-size 1024 --sample-rows 2000000 \
  --sample-seed 8301 --qmc-draws 64 --antithetic-draws

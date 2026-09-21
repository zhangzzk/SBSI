#!/usr/bin/env bash
# Three-flow photometric response comparison on their common Re>0.5 domain.
#SBATCH --job-name=sbsi_theta_old
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/theta_old_overlay_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/theta_old_overlay_v1/logs/%x_%j.err

set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
old=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $old" | sha256sum --check --status
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/sbsi_theta_mpl_${SLURM_JOB_ID}
"$python" -u -m scripts.plot_flow_theta_validation \
  --g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --sheared-catalogue "$cache/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather" \
  --response-target "$cache/mixed_shear_cde/response_target_mixed_train160_6x6x5_matrix.npz" \
  --model-10 "$base/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt" \
  --model-100 "$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt" \
  --model-old "$old" \
  --output-dir "$base/theta_old_overlay_v1/artifacts" \
  --device cuda --draws 64 --batch-size 2048 --seed 7301 --bins 10 --bootstrap 1000

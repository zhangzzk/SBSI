#!/usr/bin/env bash
# Baseline 100/100 versus uncapped intrinsic-|e| trial; second arm overlays old flow.
#SBATCH --job-name=sbsi_theta_eabs
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/%x_%A_%a.err

set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/eabs2_full_v1
baseline=$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt
trial=$root/measurement_flow_plain_eabs2_r100_t100_full_re037_s501_swaavg.pt
echo "b5fdad5486b2be781da81f3d8ef3c0f884eb9d9894bd44a0f7f9ad4863caaac8  $baseline" | sha256sum --check --status
extra=()
case ${SLURM_ARRAY_TASK_ID:?array task is required} in
  0) target=$base/response_target_mixed_train160_6x6x5_matrix_re037.npz; tag=theta_validation ;;
  1) target=$cache/mixed_shear_cde/response_target_mixed_train160_6x6x5_matrix.npz
     tag=theta_old_overlay
     old=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
     echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $old" | sha256sum --check --status
     extra+=(--model-old "$old") ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/sbsi_theta_eabs_mpl_${SLURM_JOB_ID}
"$python" -u -m scripts.plot_flow_theta_validation \
  --g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --sheared-catalogue "$cache/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather" \
  --response-target "$target" \
  --model-10 "$baseline" --label-10 '100/100 · 3D · capped' \
  --model-100 "$trial" --label-100 '100/100 · intrinsic |e| bins · uncapped' \
  "${extra[@]}" --output-dir "$root/$tag" \
  --device cuda --draws 64 --batch-size 2048 --seed 7301 --bins 10 --bootstrap 1000

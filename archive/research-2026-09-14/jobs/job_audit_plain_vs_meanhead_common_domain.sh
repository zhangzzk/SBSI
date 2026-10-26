#!/usr/bin/env bash
# Apples-to-apples derivative-loss audit on the historical Re>0.5 validation domain.

#SBATCH --job-name=sbsi_flowloss_common
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/common_domain_audit_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/common_domain_audit_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/common_domain_audit_v1
g0=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$cache/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather
response=$cache/mixed_shear_cde/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$cache/mixed_shear_e_photometric/coupling_target_mixed_validation40_6x6x5.npz

case ${SLURM_ARRAY_TASK_ID:?array task is required} in
  0)
    tag=plain_r10t10_re037
    model=$base/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt
    model_hash=0ca0a9847e17dfb0e15dc1c13df5a633ff3342c45f3e8bf53fb4b3a40c99700b
    ;;
  1)
    tag=meanhead_r500t500_re05
    model=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
    model_hash=4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73
    ;;
  2)
    tag=plain_r100t100_re037
    model=$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt
    model_hash=b5fdad5486b2be781da81f3d8ef3c0f884eb9d9894bd44a0f7f9ad4863caaac8
    ;;
  *) exit 2 ;;
esac

mkdir -p "$root/logs"
echo "$model_hash  $model" | sha256sum --check --status
output=$root/$tag.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
export PYTHONPATH=$repo
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling" \
  --output "$output" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 2000000 --sample-draws 64 --sample-seed 7301

echo "COMMON_DOMAIN_FLOW_LOSS_DONE tag=$tag output=$output"

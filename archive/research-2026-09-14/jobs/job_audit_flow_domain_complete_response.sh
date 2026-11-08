#!/usr/bin/env bash
# Audit population identity and complete-flow CRN response for pure-NLL and V3.3-like Flow E.

#SBATCH --job-name=sbsi_Edomain
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_domain_complete_response_v3/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_domain_complete_response_v3/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
photo_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
equal_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_equal_ladder_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_domain_complete_response_v3

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-1}" in
  0)
    tag=pure_nll
    model=$equal_root/measurement_flow_mixed_g0_g005_E_r0_t0_s501_swaavg.pt
    ;;
  1)
    tag=v33like_r500_t500
    model=$photo_root/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
    ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/coupling_target_mixed_validation40_6x6x5.npz
output=$root/flow_domain_complete_response_${tag}.json

mkdir -p "$root/logs"
for path in "$model" "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH=$repo
unset PYTORCH_CUDA_ALLOC_CONF

date
nvidia-smi -L
"$python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling" \
  --output "$output" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 200000 --sample-draws 64 --sample-seed 7301
date
echo "FLOW_DOMAIN_COMPLETE_RESPONSE_DONE tag=$tag output=$output"

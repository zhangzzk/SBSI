#!/usr/bin/env bash
# Matched held-out derivative-loss audit: original E and both photometric retrains.

#SBATCH --job-name=sbsi_Ephot_audit
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-2}" in
  0) tag=E_original; model=$source_root/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt ;;
  1) tag=E_r450_t500; model=$root/measurement_flow_mixed_g0_g005_E_r450_t500_s501_swaavg.pt ;;
  2) tag=E_r500_t500; model=$root/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$root/coupling_target_mixed_validation40_6x6x5.npz
output=$root/heldout_derivative_losses_${tag}_s501_v2.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for path in "$model" "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling" \
  --output "$output" --device cuda --delta 0.02

echo "MIXED_SHEAR_E_PHOTOMETRIC_AUDIT_DONE tag=$tag output=$output"

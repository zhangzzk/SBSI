#!/usr/bin/env bash
# Case-disjoint density/response/photometric audit for the coupling ladder.

#SBATCH --job-name=sbsi_Etheta_audit
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
target_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-7}" in
  0) tag=t0p5 ;;
  1) tag=t1 ;;
  2) tag=t2 ;;
  3) tag=t5 ;;
  4) tag=t10 ;;
  5) tag=t25 ;;
  6) tag=t75 ;;
  7) tag=t200 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

model=$root/measurement_flow_mixed_g0_g005_E_r450_${tag}_s501_swaavg.pt
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$target_root/coupling_target_mixed_validation40_6x6x5.npz
output=$root/heldout_derivative_losses_E_r450_${tag}_s501.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

for path in "$model" "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling" \
  --output "$output" --device cuda --delta 0.02

echo "FLOW_E_COUPLING_LADDER_AUDIT_DONE tag=$tag output=$output"

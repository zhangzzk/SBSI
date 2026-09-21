#!/usr/bin/env bash
# Independent-seed convergence audit for the strongest direct-flow response arm.

#SBATCH --job-name=sbsi_EdirectAudit
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
photo_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-3}" in
  0) draws=64;  seed=8301 ;;
  1) draws=128; seed=8301 ;;
  2) draws=256; seed=8301 ;;
  3) draws=256; seed=9301 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

model=$root/measurement_flow_plain_direct_r100_t10_d16_s501_swaavg.pt
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$photo_root/coupling_target_mixed_validation40_6x6x5.npz
output=$root/complete_response_plain_direct_r100_t10_d16_s501_d${draws}_seed${seed}.json

for path in "$model" "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

date
nvidia-smi -L
"$python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling" \
  --output "$output" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 200000 --sample-draws "$draws" --sample-seed "$seed"
date
echo "PLAIN_COMPLETE_FLOW_DRAW_AUDIT_DONE draws=$draws seed=$seed output=$output"

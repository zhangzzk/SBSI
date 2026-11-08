#!/usr/bin/env bash
# Controlled response-target comparison for the three remaining 16-draw arms.

#SBATCH --job-name=sbsi_Ed16Compare
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_grid_v1
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-2}" in
  0) tag=r5_t5_d16 ;;
  1) tag=r25_t10_d16 ;;
  2) tag=r50_t10_d16 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

model=$root/measurement_flow_plain_direct_${tag}_s501_swaavg.pt
output=$root/response_loss_gap_${tag}_s501_200k.json

mkdir -p "$root/logs"
for path in "$model" "$source_root/response_target_mixed_train160_6x6x5_matrix.npz"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

date
nvidia-smi -L
"$python" -u "$repo/scripts/diagnose_flow_response_loss_gap.py" \
  --model "$model" \
  --g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --sheared-catalogue "$source_root/det_meas_crowd_conc_g0.05_all200.feather" \
  --response-target "$source_root/response_target_mixed_train160_6x6x5_matrix.npz" \
  --output "$output" \
  --device cuda --delta 0.02 --batch-size 1024 --sample-rows 200000 \
  --sample-seed 8301 --qmc-draws 64 --antithetic-draws
date
echo "PLAIN_COMPLETE_FLOW_D16_COMPARE_DONE tag=$tag output=$output"

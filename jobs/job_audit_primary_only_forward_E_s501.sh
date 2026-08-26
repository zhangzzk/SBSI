#!/usr/bin/env bash
# Held-out primary-only forward half-shear audit for original E seed 501.

#SBATCH --job-name=sbsi_e501_primon
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde

model=$root/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz
output=$root/primary_only_forward_response_E_s501.json

for path in "$model" "$g0" "$g05" "$target"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
"$python" -u "$repo/scripts/audit_primary_only_forward_response.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$target" --output "$output" --device cuda \
  --batch-size 65536 --n-boot 10000 --bootstrap-seed 0

#!/usr/bin/env bash

#SBATCH --job-name=sbsi_mixunc
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz
output=$root/mixed_shear_response_uncertainty_all200.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for path in "$g0" "$g05" "$target"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$python" -u "$repo/scripts/audit_mixed_shear_response_uncertainty.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$target" --output "$output" \
  --n-boot 100000 --bootstrap-seed 0

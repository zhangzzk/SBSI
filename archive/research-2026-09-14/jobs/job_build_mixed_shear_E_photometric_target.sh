#!/usr/bin/env bash
# Build train-only and untouched-validation photometric derivative targets for Flow E.

#SBATCH --job-name=sbsi_Ephot_tgt
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
train_target=$root/coupling_target_mixed_train160_6x6x5.npz
validation_target=$root/coupling_target_mixed_validation40_6x6x5.npz

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for path in "$g0" "$g05" "$response"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$train_target" "$validation_target"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

"$python" -u "$repo/scripts/build_flow_coupling_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --case-role train --min-count 300 \
  --output "$train_target"
"$python" -u "$repo/scripts/build_flow_coupling_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --case-role validation --min-count 100 \
  --output "$validation_target"

echo "MIXED_SHEAR_E_PHOTOMETRIC_TARGET_DONE"

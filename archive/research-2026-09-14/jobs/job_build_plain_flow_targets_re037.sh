#!/usr/bin/env bash
# Rebuild the matched response and photometric targets for Re_input_p > 0.37 arcsec.

#SBATCH --job-name=sbsi_EtgtR037
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$root/response_target_mixed_train160_6x6x5_matrix_re037.npz
coupling_train=$root/coupling_target_mixed_train160_6x6x5_re037.npz
coupling_validation=$root/coupling_target_mixed_validation40_6x6x5_re037.npz

mkdir -p "$root/logs"
for path in "$g0" "$g05"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$response" "$coupling_train" "$coupling_validation"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

date
echo "Building matched shape-response target for true Re > 0.37 arcsec"
"$python" -u "$repo/scripts/build_flow_response_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" --output "$response" \
  --cases 200 --split-seed 501 --validation-size 0.2 \
  --n-flux 6 --n-size 6 --n-crowd 5 --min-count 500 \
  --primary-mag-max 25.8 --primary-re-min 0.37

echo "Building train and untouched-validation photometric targets on the same grid/domain"
"$python" -u "$repo/scripts/build_flow_coupling_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --case-role train --min-count 300 \
  --output "$coupling_train"
"$python" -u "$repo/scripts/build_flow_coupling_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --case-role validation --min-count 100 \
  --output "$coupling_validation"
date
echo "PLAIN_FLOW_RE037_TARGETS_DONE response=$response"

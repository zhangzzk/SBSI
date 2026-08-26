#!/usr/bin/env bash
# Held-out full-matrix audit for trained C/D/E checkpoints (array 0/1/2).

#SBATCH --job-name=sbsi_mixaudit
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=6
# The partition's 16 GB GPU nodes expose at most ~41 GB host RAM.
# Six million held-out rows fit within 40 GB when loaded with this compact column set.
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
arms=(C D E)
arm=${arms[${SLURM_ARRAY_TASK_ID:?submit as array 0-2}]}

model=$root/measurement_flow_mixed_g0_g005_${arm}_s501_swaavg.pt
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz
output=$root/heldout_response_matrix_${arm}_s501.json
export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for path in "$model" "$g0" "$g05" "$target"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$python" -u "$repo/scripts/audit_flow_response_matrix.py" \
  --model "$model" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$target" --output "$output" --device cuda --delta 0.02

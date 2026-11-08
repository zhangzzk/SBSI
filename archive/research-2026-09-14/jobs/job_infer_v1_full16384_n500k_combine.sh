#!/usr/bin/env bash
# Combine the paired A40/V100 16,384-draw partitions by exact moment summation.

#SBATCH --job-name=sbsi_full16384_n500k_combine
#SBATCH --partition=inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
output="$root/combined"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
export PYTHONPATH="$repo"

"$python" "$repo/scripts/combine_numerical_recenter_partitions.py" \
  --part "$root/results/observations_000000_249999_a40" \
  --part "$root/results/observations_250000_499999_v100" \
  --output "$output"

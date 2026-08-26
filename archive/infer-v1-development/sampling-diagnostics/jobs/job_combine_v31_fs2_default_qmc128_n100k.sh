#!/usr/bin/env bash
# Archived pre-Infer-V1 100k result combiner.

#SBATCH --job-name=sbsi_fs2_n100k_combine
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export PYTHONPATH="$repo"

"$python" "$repo/scripts/combine_numerical_recenter_partitions.py" \
  --part "$root/results/observations_000000_049999" \
  --part "$root/results/observations_050000_099999" \
  --output "$root/combined"

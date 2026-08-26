#!/usr/bin/env bash
# Combine the two one-million-closure partitions by exact moment summation.

#SBATCH --job-name=sbsi_infer_v1_n1m_combine
#SBATCH --partition=cip
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export PYTHONPATH="$repo"

"$python" "$repo/scripts/combine_numerical_recenter_partitions.py" \
  --part "$root/results/observations_000000_499999" \
  --part "$root/results/observations_500000_999999" \
  --output "$root/combined"

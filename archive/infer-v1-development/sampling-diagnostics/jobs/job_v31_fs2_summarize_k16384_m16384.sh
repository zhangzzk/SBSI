#!/usr/bin/env bash
# Archived Infer V1 sampling diagnostic job.
# Summarize the paired K=16,384 experiment after both proposal seeds complete.

#SBATCH --job-name=sbsi_k16384_summary
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

"$python" "$repo/scripts/summarize_k16384_m16384_diagnostic.py" \
  --root "$root" \
  --output "$root/k16384_m16384_summary.json"

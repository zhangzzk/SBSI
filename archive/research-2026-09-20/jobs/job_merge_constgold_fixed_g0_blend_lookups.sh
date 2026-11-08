#!/usr/bin/env bash
# Merge and validate the five disjoint lookup blocks for each emulator.

#SBATCH --job-name=sbsi_cgfg0_rbmerge
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
cd "$repo"

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
for version in v1 v2; do
  inputs=()
  for first in 40 50 60 70 80; do
    last=$((first + 9))
    inputs+=(--input "$root/rblend_${version}/blocks/cases${first}_${last}.feather")
  done
  "$python" -u -m scripts.merge_constgold_fixed_g0_blend_lookups \
    "${inputs[@]}" "${cases[@]}" \
    --output "$root/rblend_${version}_cases40_89.feather"
done

echo "CONSTGOLD_FIXED_G0_RBLEND_MERGE_DONE root=$root"

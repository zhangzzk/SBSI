#!/usr/bin/env bash
# Merge the per-block R_blend lookups for both arms of the rejection A/B.

#SBATCH --job-name=sbsi_cgfg0_rb_ab_merge
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_rejection_ab_v1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
for version in v2_baseline v2_relaxed; do
  inputs=()
  for first in 40 50 60 70 80; do
    last=$((first + 9))
    inputs+=(--input "$root/rblend_${version}/blocks/cases${first}_${last}.feather")
  done
  "$python" -u -m scripts.merge_constgold_fixed_g0_blend_lookups \
    "${inputs[@]}" "${cases[@]}" \
    --output "$root/rblend_${version}_cases40_89.feather"
  echo "MERGED version=$version"
done

echo "CONSTGOLD_RB_AB_MERGE_DONE root=$root"

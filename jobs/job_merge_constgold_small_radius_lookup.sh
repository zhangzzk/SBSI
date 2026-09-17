#!/usr/bin/env bash
# Merge the six disjoint R_blend blocks used by the small-radius diagnosis.

#SBATCH --job-name=sbsi_cgrad_merge
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_small_radius_c40_139_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_small_radius_c40_139_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
property=$v2/constgold_property_bias_c40_139_v1
root=$v2/constgold_small_radius_c40_139_v1
output=$root/rblend_cases40_139.feather

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done
inputs=(
  --input "$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather"
)
for first in 90 100 110 120 130; do
  last=$((first + 9))
  inputs+=(--input "$property/rblend/blocks/cases${first}_${last}.feather")
done

"$python" -u -m scripts.merge_constgold_fixed_g0_blend_lookups \
  "${inputs[@]}" "${cases[@]}" --output "$output"

echo "CONSTGOLD_SMALL_RADIUS_LOOKUP_DONE output=$output"

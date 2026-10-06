#!/usr/bin/env bash
# Measure property-binned m for all three usability branches on cases40--139.

#SBATCH --job-name=sbsi_cgprop
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_property_bias_c40_139_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_property_bias_c40_139_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
root=$v2/constgold_property_bias_c40_139_v1
old_lookup=$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather
reference=$v2/constgold_response_c40_89_v1/result.json
figure=$repo/plots/figures/constgold_property_bias_c40_139

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done
classifiers=()
for seed in 20260913 20260914 20260915; do
  classifiers+=(--classifier "$v1/classifier/seed${seed}/selected.pt")
done
rblend=(--rblend "$old_lookup")
for first in 90 100 110 120 130; do
  last=$((first + 9))
  rblend+=(--rblend "$root/rblend/blocks/cases${first}_${last}.feather")
done

"$python" -u plots/constgold_property_bias.py compute \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$v2/flow/paired/selected.pt" \
  "${rblend[@]}" "${classifiers[@]}" "${cases[@]}" \
  --result "$root/result.json" \
  --bins 8 --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260914 --device cuda \
  --reference-result "$reference" --reference-model new_v2_staged

"$python" -u plots/constgold_property_bias.py plot \
  --result "$root/result.json" \
  --output-prefix "$figure"

echo "CONSTGOLD_PROPERTY_BIAS_DONE result=$root/result.json figure=$figure"

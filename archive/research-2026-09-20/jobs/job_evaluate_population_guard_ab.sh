#!/usr/bin/env bash
# Frozen diagnostic: baseline/shared/independent on uncut and fixed-cut anchors.
#SBATCH --job-name=sbsi_popguard_cg
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/population_guard_ab_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/population_guard_ab_20260917_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
full=$cache/full_domain_true_mag26_guard_v1
arms=(baseline shared independent)
arm=${arms[$((SLURM_ARRAY_TASK_ID % 3))]}
flow=$root/$arm/selected.pt
if [[ $arm == baseline ]]; then
  flow=$full/flow/guard/selected.pt
fi
rblend=()
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  cohort=uncut
  anchor=$full/domain/flow/g0/case{case:03d}.npz
  lookups=$cache/full_domain_true_mag26_guard_refine_v1/constgold_property_bias_uncut_true_mag26_c40_139_v1/rblend/blocks
  for first in $(seq 40 10 130); do
    rblend+=(--rblend "$lookups/cases${first}_$((first + 9)).feather")
  done
else
  cohort=fixed_cut
  anchor=$v2/domain/flow/g0/case{case:03d}.npz
  rblend+=(--rblend "$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather")
  for first in 90 100 110 120 130; do
    rblend+=(--rblend "$v2/constgold_property_bias_c40_139_v1/rblend/blocks/cases${first}_$((first + 9)).feather")
  done
fi
classifiers=()
for seed in 20260913 20260914 20260915; do
  classifiers+=(--classifier "$v1/classifier/seed${seed}/selected.pt")
done
cases=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done
output=$root/constgold/$cohort/$arm.json
[[ ! -e $output ]] || { echo "refusing to overwrite $output" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -u -m plots.constgold_property_bias compute \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$anchor" --flow "$flow" --model-label "Population-guard $arm $cohort" \
  --true-magnitude-max 26 "${rblend[@]}" "${classifiers[@]}" "${cases[@]}" \
  --result "$output" --bins 8 --h 0.02 --draws 64 --sampling-seed 7301 \
  --batch-size 1024 --n-boot 10000 --bootstrap-seed 20260914 --device cuda

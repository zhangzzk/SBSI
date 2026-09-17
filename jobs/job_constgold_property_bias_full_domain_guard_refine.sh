#!/usr/bin/env bash
# Recompute the 100-case property panels for the sharp-guard full-domain flow.

#SBATCH --job-name=sbsi_cgprop_refine
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_property_bias_true_mag26_c40_139_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_property_bias_true_mag26_c40_139_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
refine=$cache/full_domain_true_mag26_guard_refine_v1
root=$refine/constgold_property_bias_true_mag26_c40_139_v2
flow=$refine/flow/selected.pt
old_lookup=$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather
figure=$repo/plots/figures/constgold_property_bias_full_domain_guard_refine_true_mag26_c40_139_v2

if [[ -e "$root/result.json" || -e "${figure}.png" || -e "${figure}.pdf" ]]; then
  echo "refusing to overwrite existing sharp-guard property result" >&2
  exit 1
fi
mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L

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
  rblend+=(--rblend "$v2/constgold_property_bias_c40_139_v1/rblend/blocks/cases${first}_${last}.feather")
done

"$python" -u -m plots.constgold_property_bias compute \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$flow" --model-label "Sharp-guard full-domain flow" \
  --true-magnitude-max 26 \
  "${rblend[@]}" "${classifiers[@]}" "${cases[@]}" \
  --result "$root/result.json" \
  --bins 8 --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260914 --device cuda

"$python" -u -m plots.constgold_property_bias plot \
  --result "$root/result.json" \
  --output-prefix "$figure"

echo "CONSTGOLD_REFINED_PROPERTY_BIAS_DONE result=$root/result.json figure=$figure"

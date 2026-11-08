#!/usr/bin/env bash
# Recompute the 100-case property panels for both 6x6x6 grid-flow variants.

#SBATCH --job-name=sbsi_cgprop_grid
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
variants=(truth measured)
labels=("Truth-property 6x6x6 grid flow" "Measured-anchor 6x6x6 grid flow")
variant=${variants[${SLURM_ARRAY_TASK_ID:?}]}
label=${labels[${SLURM_ARRAY_TASK_ID}]}
root=$v2/constgold_property_bias_grid_${variant}_c40_139_v1
flow=$v2/grid_flows_v1/$variant/training/selected.pt
old_lookup=$v2/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather
figure=$repo/plots/figures/constgold_property_bias_grid_${variant}_c40_139

if [[ -e "$root/result.json" || -e "${figure}.png" || -e "${figure}.pdf" ]]; then
  echo "refusing to overwrite existing grid-flow property result" >&2
  exit 1
fi
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
  rblend+=(--rblend "$v2/constgold_property_bias_c40_139_v1/rblend/blocks/cases${first}_${last}.feather")
done

"$python" -u plots/constgold_property_bias.py compute \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --flow "$flow" --model-label "$label" \
  "${rblend[@]}" "${classifiers[@]}" "${cases[@]}" \
  --result "$root/result.json" \
  --bins 8 --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260914 --device cuda

"$python" -u plots/constgold_property_bias.py plot \
  --result "$root/result.json" \
  --output-prefix "$figure"

echo "CONSTGOLD_GRID_PROPERTY_BIAS_DONE variant=$variant result=$root/result.json figure=$figure"

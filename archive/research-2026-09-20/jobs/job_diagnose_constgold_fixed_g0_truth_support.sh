#!/usr/bin/env bash
# Localize fixed-g0 matched-response bias inside/outside the former truth support.

#SBATCH --job-name=sbsi_cgfg0_domain
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/domain_diagnosis_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
historical=$cache/plain_complete_flow_response_re037_v1
historical_run=$historical/physical_circularized_grid_free_shape_radius_lambda10_v2
historical_lookup=$historical/constgold50_trial9_base_lambda05_epoch060_v1

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done

historical_rblend=$root/historical_v35_rblend_c40_89.feather
if [[ ! -f "$historical_rblend" ]]; then
  merge_inputs=()
  for block in $(seq 0 4); do
    merge_inputs+=(--input "$historical_lookup/lookup_${block}.feather")
  done
  "$python" -u -m scripts.merge_constgold_fixed_g0_blend_lookups \
    "${merge_inputs[@]}" "${cases[@]}" \
    --output "$historical_rblend"
fi

"$python" -u -m scripts.diagnose_constgold_fixed_g0_truth_support \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --model "historical_v35=$historical_run/constgold50_v1/selected_epoch154.pt" \
  --model "old_fixed_v1_nll=$v1/flow/nll/selected.pt" \
  --model "new_v2_staged=$v2/flow/paired/selected.pt" \
  --rblend "historical_v35=$historical_rblend" \
  --rblend "old_fixed_v1_nll=$root/../rblend_v1_cases40_89.feather" \
  --rblend "new_v2_staged=$root/../rblend_v2_cases40_89.feather" \
  "${cases[@]}" \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260915 --device cuda

echo "CONSTGOLD_TRUTH_SUPPORT_JOB_DONE output=$root/result.json"

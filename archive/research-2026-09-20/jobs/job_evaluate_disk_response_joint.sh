#!/usr/bin/env bash
#SBATCH --job-name=sbsi_disk_joint
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/disk_response_moment_20260918_v1
parent=$cache/constgold_truth_parent_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
model=/project/ls-gruen/users/zekang.zhang/blendemu_runs/disk_response_moment_20260918_v1/model
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mode=${1:?smoke or run}
if [[ $mode == smoke ]]; then
  cases=(40 41)
  options=(--rblend "$cache/output_conditioned_response_20260917_v1/joint_smoke_rblend.feather")
  output=$root/joint_smoke.json
elif [[ $mode == run ]]; then
  mapfile -t cases < <(seq 40 119)
  options=()
  for first in $(seq 40 10 110); do
    options+=(--rblend "$parent/rblend/cases${first}_$((first+9)).feather")
  done
  output=$root/joint_full.json
else
  exit 2
fi
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
"$python" -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$parent/parent_v2" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --response-candidate "$model" --response-composition mobius \
  --pair-root "$cache/output_conditioned_response_20260917_v1/pairs" \
  --response-threads 8 --response-device cuda \
  "${options[@]}" --output "$output" --cases "${cases[@]}"
if [[ $mode == run ]]; then
  "$python" -u -m scripts.compare_truth_parent_response \
    --baseline "$parent/baseline_response.json" --candidate "$output" --output "$root/joint_comparison.json"
fi

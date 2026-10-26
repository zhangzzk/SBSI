#!/usr/bin/env bash
#SBATCH --job-name=sbsi_guard_cv_joint
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_control_variate_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/guard_control_variate_20260917_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/guard_control_variate_20260917_v1
parent=$cache/constgold_truth_parent_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
step=$("$python" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["full_density_gate_passed"]; print(x["best_step"])' "$root/refinement/completed.json")
if [[ $step == 0 ]]; then
  echo "JOINT_DIAGNOSTIC_SKIPPED unchanged original checkpoint selected; baseline already evaluated"
  exit 0
fi
mapfile -t cases < <(seq 40 119)
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
for first in $(seq 40 10 110); do
  options+=(--rblend "$parent/rblend/cases${first}_$((first+9)).feather")
done
"$python" -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$parent/parent_v2" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --flow "$root/refinement/selected.pt" \
  "${options[@]}" --output "$root/refinement_truth_parent_response.json" --cases "${cases[@]}"
"$python" -u -m scripts.compare_truth_parent_response \
  --baseline "$parent/baseline_response.json" --candidate "$root/refinement_truth_parent_response.json" \
  --output "$root/refinement_truth_parent_comparison.json"

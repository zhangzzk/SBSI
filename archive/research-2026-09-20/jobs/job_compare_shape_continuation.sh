#!/usr/bin/env bash
#SBATCH --job-name=sbsi_shape_compare
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
endpoint=${1:?main or independent}
case $endpoint in main|independent) ;; *) exit 2 ;; esac
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" -u -m scripts.compare_truth_parent_response \
  --baseline "$root/control/joint_${endpoint}_full.json" \
  --candidate "$root/augmented/joint_${endpoint}_full.json" \
  --output "$root/${endpoint}_augmentation_response_comparison.json"
"$python" -u -m scripts.compare_joint_predictive_results \
  --baseline "$root/control/joint_${endpoint}_full.json" \
  --candidate "$root/augmented/joint_${endpoint}_full.json" \
  --output "$root/${endpoint}_augmentation_density_comparison.json"
if [[ $endpoint == independent ]]; then
  for arm in control augmented; do
    for kind in response density; do
      if [[ $kind == response ]]; then module=scripts.compare_truth_parent_response; else module=scripts.compare_joint_predictive_results; fi
      "$python" -u -m "$module" \
        --baseline /project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/independent_candidate_full.json \
        --candidate "$root/$arm/joint_independent_full.json" \
        --output "$root/${arm}/independent_vs_initial_${kind}.json"
    done
  done
else
  for arm in control augmented; do
    "$python" -u -m scripts.compare_truth_parent_response \
      --baseline "$root/initial_main_truth_parent.json" \
      --candidate "$root/$arm/joint_main_full.json" \
      --output "$root/$arm/main_vs_initial_response.json"
  done
fi

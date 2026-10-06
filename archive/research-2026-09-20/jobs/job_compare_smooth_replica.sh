#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_replica_compare
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
for endpoint in fresh_g1 fresh_g2 main independent; do
 "$python" -u -m scripts.compare_truth_parent_response \
  --baseline "$root/smooth_joint_${endpoint}.json" \
  --candidate "$root/replica_results/smooth_joint_${endpoint}.json" \
  --output "$root/replica_results/smooth_training_repeat_${endpoint}.json"
done
args=()
for endpoint in fresh_g1 fresh_g2 main independent; do
 args+=(--result "$root/replica_results/smooth_joint_${endpoint}.json")
 args+=(--receipt "$root/replica_results/smooth_joint_${endpoint}_comparison.json")
done
exec "$python" -u -m scripts.summarize_smooth_classifier_stability "${args[@]}" \
 --output "$root/replica_results/smooth_full_influence.json"

#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_influence
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
"$python" -m pytest tests/test_smooth_classifier_stability.py -q
arguments=()
for endpoint in fresh_g1 fresh_g2 main independent; do
 arguments+=(--result "$root/smooth_joint_${endpoint}.json")
 arguments+=(--receipt "$root/smooth_joint_${endpoint}_comparison.json")
done
exec "$python" -u -m scripts.summarize_smooth_classifier_stability "${arguments[@]}" \
 --output "$root/smooth_full_influence.json"

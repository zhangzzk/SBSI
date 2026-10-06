#!/usr/bin/env bash
#SBATCH --job-name=sbsi_initial_main_compare
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
for arm in control augmented; do
  "$python" -u -m scripts.compare_truth_parent_response \
    --baseline "$root/initial_main_truth_parent.json" \
    --candidate "$root/$arm/joint_main_full.json" \
    --output "$root/$arm/main_vs_initial_response.json"
done

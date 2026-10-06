#!/usr/bin/env bash
#SBATCH --job-name=sbsi_selected_density_normalization
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
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
"$python" -m pytest tests/test_selected_density_normalization.py -q
exec "$python" -u -m scripts.normalize_selected_density_comparison \
 --reference "$root/initial_main_truth_parent.json" \
 --candidate "$root/joint_probability_refinement_full/joint_main_full.json" \
 --density "$root/joint_probability_refinement_full/transported_density_full.json" \
 --output "$root/joint_probability_refinement_full/normalized_selected_density.json"

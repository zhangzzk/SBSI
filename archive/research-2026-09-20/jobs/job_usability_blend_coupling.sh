#!/usr/bin/env bash
#SBATCH --job-name=sbsi_usability_coupling
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
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
"$python" -m pytest tests/test_usability_blend_coupling.py -q
args=()
for file in joint_main_full joint_independent_full fresh_g1_full fresh_g2_full; do
 args+=(--result "$base/joint_probability_refinement_full/$file.json")
done
exec "$python" -u -m scripts.analyze_usability_blend_coupling "${args[@]}" --output "$base/usability_blend_coupling.json"

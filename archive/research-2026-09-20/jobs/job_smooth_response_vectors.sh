#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_vectors
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" -m pytest tests/test_smooth_response_vectors.py -q
exec "$PY" -u -m scripts.summarize_smooth_response_vectors \
 --root /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1 \
 --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/smooth_response_vectors.json

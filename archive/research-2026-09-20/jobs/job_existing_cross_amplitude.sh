#!/usr/bin/env bash
#SBATCH --job-name=sbsi_cross_amplitude
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_existing_cross_amplitude \
 --prepared-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/full \
 --result-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1 \
 --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/existing_cross_amplitude.json

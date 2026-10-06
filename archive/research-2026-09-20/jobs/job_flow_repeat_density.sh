#!/usr/bin/env bash
#SBATCH --job-name=sbsi_flow_repeat_density
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" -m pytest tests/test_flow_repeat_density.py tests/test_selected_density_normalization.py -q
exec "$PY" -u -m scripts.audit_flow_repeat_density \
 --protocol /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/flow_refinement_repeat_density_protocol.json \
 --index "$SLURM_ARRAY_TASK_ID"

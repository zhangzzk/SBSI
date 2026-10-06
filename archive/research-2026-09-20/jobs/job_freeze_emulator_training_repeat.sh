#!/usr/bin/env bash
#SBATCH --job-name=sbsi_freeze_emulator_repeat
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" -m pytest tests/test_emulator_training_repeat.py -q
exec "$PY" -u -m scripts.validate_emulator_training_repeat \
 --protocol /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/emulator_repeat_joint_template.json \
 --freeze-output /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/emulator_repeat_joint_protocol.json

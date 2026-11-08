#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_freeze
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
"$python" -m pytest tests/test_smooth_classifier_joint.py tests/test_smooth_classifier_features.py -q
exec "$python" -u -m scripts.freeze_smooth_classifier --template "${1:?template}" --output "${2:?protocol}"

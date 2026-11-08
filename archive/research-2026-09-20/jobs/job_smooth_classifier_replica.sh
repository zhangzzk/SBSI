#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_replica
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
"$python" -m pytest tests/test_smooth_replica_contract.py -q
exec "$python" -u -m scripts.train_smooth_classifier_replica \
 --frozen-protocol "$root/smooth_replica_training_protocol.json" \
 --prepared-root "$root/smooth_prepared_full" --output-root "$root/smooth_classifier_seed20260914" \
 --arm smooth --device cpu

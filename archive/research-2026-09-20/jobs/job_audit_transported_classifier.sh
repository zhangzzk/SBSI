#!/usr/bin/env bash
#SBATCH --job-name=sbsi_transported_classifier_audit
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
template=${1:?predeclared validation template}
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
"$python" -m pytest tests/test_freeze_transported_classifier.py tests/test_scene_classifier_calibration.py -q
"$python" -u -m scripts.freeze_transported_classifier --template "$template" --output "$root/transported_calibration_protocol.json"
exec "$python" -u -m scripts.audit_transported_classifier_calibration \
 --protocol "$root/transported_calibration_protocol.json" --output "$root/transported_calibration_full.json"

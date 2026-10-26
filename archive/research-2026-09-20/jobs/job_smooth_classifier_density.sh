#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_density
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
mode=${1:?collect or full}
suffix=$mode
if [[ "$mode" == collect ]]; then suffix=observed; fi
"$python" -m pytest tests/test_smooth_classifier_density.py -q
exec "$python" -u -m scripts.audit_smooth_classifier_density \
 --protocol "$root/smooth_density_protocol.json" --mode "$mode" \
 --observed "$root/smooth_density_observed.json" --output "$root/smooth_density_${suffix}.json"

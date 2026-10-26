#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_classifier
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
stage=${1:?prepare or train}
scope=${2:?smoke or full}
args=()
case "$scope" in
 smoke) args+=(--smoke) ;;
 full) test -f "$root/smooth_classifier_smoke/completed.json" ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_smooth_classifier_features.py tests/test_continuous_crowding_residuals.py -q
case "$stage" in
 prepare) exec "$python" -u -m scripts.prepare_smooth_classifier \
  --protocol "$root/smooth_classifier_protocol.json" \
  --output-root "$root/smooth_prepared_${scope}" "${args[@]}" ;;
 train) exec "$python" -u -m scripts.train_smooth_classifier \
  --frozen-protocol "$root/smooth_classifier_protocol.json" \
  --prepared-root "$root/smooth_prepared_${scope}" \
  --output-root "$root/smooth_classifier_${scope}" --arm smooth --device cpu "${args[@]}" ;;
 *) exit 2 ;;
esac

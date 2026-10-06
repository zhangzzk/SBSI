#!/usr/bin/env bash
#SBATCH --job-name=sbsi_coherent_classifier
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
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
scope=${1:?smoke or full}
protocol=${2:?protocol path}
args=()
case "$scope" in
 smoke) args+=(--smoke) ;;
 full) test -f "$base/coherent_classifier_smoke.json" ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_coherent_classifier_response.py -q
exec "$python" -u -m scripts.audit_coherent_classifier_response \
 --protocol "$protocol" --output "$base/coherent_classifier_${scope}.json" "${args[@]}"

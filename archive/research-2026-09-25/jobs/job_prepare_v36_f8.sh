#!/bin/bash
#SBATCH --job-name=sbsi-f8-prepare
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v36_f8_prepare_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches
ROOT="$CACHE/v36_f8_20260924_v1"
"$PY" -m pytest tests/test_v36_feature_projection.py -q
"$PY" scripts/project_v36_flow_features.py \
  --source "$CACHE/full_domain_closecrowd_20260924_v1/domain" \
  --output "$ROOT/base/domain" --workers 4
"$PY" scripts/project_v36_flow_features.py \
  --source "$CACHE/harmonized_closecrowd_20260924_v1/domain" \
  --target-banks "$CACHE/flow_joint_closecrowd_20260924_v1" \
  --output "$ROOT/harmonized/domain" --workers 4

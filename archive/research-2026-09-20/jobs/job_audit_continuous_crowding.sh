#!/usr/bin/env bash
#SBATCH --job-name=sbsi_continuous_crowding
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
"$python" -m pytest tests/test_continuous_crowding_residuals.py -q
exec "$python" -u -m scripts.audit_continuous_crowding_residuals \
 --protocol "${1:?frozen audit protocol}" --output "${2:?new output}"

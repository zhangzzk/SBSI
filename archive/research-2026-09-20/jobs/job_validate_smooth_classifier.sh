#!/usr/bin/env bash
#SBATCH --job-name=sbsi_smooth_joint
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.validate_smooth_classifier \
 --protocol "${1:?frozen validation protocol}" --index "$SLURM_ARRAY_TASK_ID"

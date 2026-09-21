#!/usr/bin/env bash
#SBATCH --job-name=sbsi_per_leg_joint
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
phase=${1:?existing or fresh}
protocol=${2:?frozen downstream protocol path required}
index=${SLURM_ARRAY_TASK_ID:?two-task array required}
if (( index < 0 || index > 1 )); then exit 2; fi
case $phase in
  existing) endpoints=(main independent) ;;
  fresh) endpoints=(fresh1 fresh2) ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.validate_per_leg_candidate \
  "${endpoints[$index]}" --protocol \
  "$protocol"

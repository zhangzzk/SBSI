#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_bind
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.validate_independent_joint \
 --protocol "$root/joint_template.json" --freeze-output "$root/joint_protocol.json"

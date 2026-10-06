#!/usr/bin/env bash
#SBATCH --job-name=sbsi_response_tensor
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_response_label_tensor \
  --prepared /project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1/prepared \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/label_tensor.json

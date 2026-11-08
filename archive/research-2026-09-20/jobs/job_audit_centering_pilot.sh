#!/usr/bin/env bash
#SBATCH --job-name=sbsi_centering_audit
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_common_optimizer_pilot \
  --pilot-root /project/ls-gruen/users/zekang.zhang/blendemu_runs/common_optimizer_subpixel_pilot_20260917_v1 \
  --reference-pilot /project/ls-gruen/users/zekang.zhang/blendemu_runs/common_optimizer_primary_pilot_20260917_v1 \
  --cases 0 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/centering_pilot.json

#!/usr/bin/env bash
#SBATCH --job-name=sbsi_harmonized_density
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_harmonized_flow_density

#!/bin/bash
#SBATCH --job-name=target-response-se
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/target_response_uncertainty_20261104/job-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /project/ls-gruen/users/zekang.zhang/sbsi_caches/target_response_uncertainty_20261104/audit.py

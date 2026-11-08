#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=seed_response_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_seed_response_20261107/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I /home/z/Zekang.Zhang/SBSI/doc/evidence/measured_seed_response_20261107/report.py

#!/bin/bash
#SBATCH --job-name=primary_boundary_report
#SBATCH --partition=cluster
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/primary_selection_transport_20261107/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/primary_selection_transport_20261107/report.py

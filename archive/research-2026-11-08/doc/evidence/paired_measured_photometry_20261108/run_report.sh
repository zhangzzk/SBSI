#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=paired_photo_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/paired_measured_photometry_20261108/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/paired_measured_photometry_20261108/report.py

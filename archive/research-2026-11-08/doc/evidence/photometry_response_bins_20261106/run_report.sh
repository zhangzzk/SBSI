#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=photo_bins_report
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/photometry_response_bins_20261106/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/photometry_response_bins_20261106/report.py "$@"

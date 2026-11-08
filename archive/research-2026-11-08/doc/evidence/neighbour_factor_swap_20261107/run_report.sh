#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=factor_swap_report
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_factor_swap_20261107/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_factor_swap_20261107/pipeline.py report

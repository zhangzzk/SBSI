#!/bin/bash
#SBATCH --partition=inter,cip
#SBATCH --job-name=current_measured_pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/current_measured_photometry_20261107/logs/pilot_%A_%a.log
#SBATCH --gres=gpu:a40-16gb:1
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/current_measured_photometry_20261107/evaluate.py pilot

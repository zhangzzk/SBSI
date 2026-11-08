#!/bin/bash
#SBATCH --partition=inter,cip
#SBATCH --job-name=minimal_numerical
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_flow_20261108/logs/minimal_numerical_%A_%a.log
#SBATCH --gres=gpu:a40-16gb:1
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/minimal_flow_20261108/pilot.py

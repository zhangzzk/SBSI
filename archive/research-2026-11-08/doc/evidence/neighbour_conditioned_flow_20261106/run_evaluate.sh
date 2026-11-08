#!/bin/bash
#SBATCH --partition=inter,cip
#SBATCH --job-name=neighbour_flow_constant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_conditioned_flow_20261106/logs/evaluate_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_conditioned_flow_20261106/evaluate.py

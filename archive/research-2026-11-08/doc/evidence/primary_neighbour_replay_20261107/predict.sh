#!/bin/bash
#SBATCH --partition=cip,inter
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --job-name=primary_model
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107/logs/%x_%j.out
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
exec /project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/primary_neighbour_replay_20261107/predict.py

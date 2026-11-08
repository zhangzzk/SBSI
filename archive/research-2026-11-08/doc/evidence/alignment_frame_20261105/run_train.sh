#!/bin/bash
#SBATCH --job-name=alignment_frame
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u \
 /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_frame_20261105/train.py

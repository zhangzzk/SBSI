#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=neighbour_crossing_shapes
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:12:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107/logs/%x_%j.out
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/neighbour_crossing_response_20261107/cross.py

#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=measured_response_symmetry
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_response_symmetry_20261107/audit_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/measured_response_symmetry_20261107/audit.py

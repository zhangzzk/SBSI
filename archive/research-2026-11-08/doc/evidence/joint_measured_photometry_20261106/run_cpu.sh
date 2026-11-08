#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=joint_measured_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_measured_photometry_20261106/logs/cpu_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u "/home/z/Zekang.Zhang/SBSI/doc/evidence/joint_measured_photometry_20261106/$1.py" "${@:2}"

#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=guard_compare_report
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_regularized_20261106/report_constant.py "${1:?pilot, interim or full}"

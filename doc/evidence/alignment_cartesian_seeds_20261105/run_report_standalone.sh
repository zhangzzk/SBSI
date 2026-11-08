#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=seed_standalone_report
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_cartesian_seeds_20261105/logs/report_%j.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/alignment_cartesian_seeds_20261105/report_standalone.py

#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=component-report
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constant_component_accounting_20261104/job-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python /project/ls-gruen/users/zekang.zhang/sbsi_caches/constant_component_accounting_20261104/report.py

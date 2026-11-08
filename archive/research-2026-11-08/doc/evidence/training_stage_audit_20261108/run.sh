#!/bin/bash
#SBATCH --partition=inter
#SBATCH --job-name=training_stage_audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/training_stage_audit_20261108/report_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/training_stage_audit_20261108/reduce.py

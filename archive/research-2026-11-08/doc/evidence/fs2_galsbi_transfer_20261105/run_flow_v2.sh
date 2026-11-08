#!/bin/bash
#SBATCH --job-name=fs2_galsbi_transfer
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105/logs/%x_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
family=fs2
if [[ ${SLURM_ARRAY_TASK_ID} == 1 ]]; then family=galsbi; fi
cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u \
 /home/z/Zekang.Zhang/SBSI/doc/evidence/fs2_galsbi_transfer_20261105/flow_compare_v2.py "$family"

#!/bin/bash
#SBATCH --job-name=target-constant
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_target_guard_constant_20261104/logs/%x_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
R=/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_target_guard_constant_20261104
cd /tmp
if [[ "$1" == draws ]]; then
 /project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I -u "$R/source/run.py" draws "${SLURM_ARRAY_TASK_ID:?}"
else
 /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u "$R/source/run.py" summary
fi

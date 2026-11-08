#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=measured_seed_response
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_seed_response_20261107/%x_%A_%a.log
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/blendemu_galsbi_resim
if [[ "$1" == pilot ]]; then
 /project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python -I /home/z/Zekang.Zhang/SBSI/doc/evidence/measured_seed_response_20261107/report.py --prepare
fi
if [[ "$1" == worker ]]; then
 python -u scripts/diagnose_constant_seed_response.py worker --group "${SLURM_ARRAY_TASK_ID:?}"
else
 python -u scripts/diagnose_constant_seed_response.py "$@"
fi

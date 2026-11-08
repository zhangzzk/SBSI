#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=selected_seed_calibration
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --constraint=fast-io&x86-64-v3
#SBATCH --exclude=th-cl-naples06,th-cl-naples24
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/selected_seed_calibration_20261108/%x_%A_%a.log
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/blendemu_galsbi_resim
if [[ "$1" == worker ]]; then
 python -u scripts/diagnose_selected_seed_calibration.py worker --group "${SLURM_ARRAY_TASK_ID:?}"
else
 python -u scripts/diagnose_selected_seed_calibration.py "$@"
fi

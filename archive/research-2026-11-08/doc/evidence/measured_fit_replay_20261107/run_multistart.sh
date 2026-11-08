#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=measurement_multistart
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_fit_replay_20261107/multistart_%j.log
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/blendemu_galsbi_resim
python -u scripts/diagnose_constant_multistart.py

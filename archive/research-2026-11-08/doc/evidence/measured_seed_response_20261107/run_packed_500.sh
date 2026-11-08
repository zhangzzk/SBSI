#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=seed_response_packed
#SBATCH --cpus-per-task=60
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --constraint=fast-io&x86-64-v3
#SBATCH --exclude=th-cl-naples06,th-cl-naples24
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_seed_response_20261107/packed_%A_%a.log
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/blendemu_galsbi_resim
python -u scripts/run_seed_response_packed.py

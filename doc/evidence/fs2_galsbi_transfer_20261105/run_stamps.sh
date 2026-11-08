#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=transfer_stamps
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105/logs/stamps_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python -I -u /home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/audit_population_stamp_transfer.py

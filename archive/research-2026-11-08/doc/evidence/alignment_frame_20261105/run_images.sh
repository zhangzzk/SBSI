#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --job-name=identical_galaxies
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_frame_20261105/logs/images_%A_%a.log
set -euo pipefail
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd /tmp
/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python -I -u /home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/audit_identical_galaxy_conditions.py

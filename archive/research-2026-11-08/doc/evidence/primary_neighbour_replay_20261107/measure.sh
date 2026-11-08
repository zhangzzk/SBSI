#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=01:30:00
#SBATCH --job-name=primary_measure
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107/logs/%x_%j.out
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; else source /usr/share/Modules/init/bash; fi
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
exec python -u /home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/measure_primary_replay.py "$@"

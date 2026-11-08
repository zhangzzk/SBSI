#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=32
#SBATCH --mem=115G
#SBATCH --time=10:00:00
#SBATCH --job-name=primary_fields
#SBATCH --array=0-3%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107/logs/%x_%A_%a.out
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
source /etc/profile.d/modules.sh
module load sextractor
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
primary_root=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107
primary_tmp=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
# Per worker: private native float32 image plus full uncompressed SExtractor catalogue.
if (( $(df --output=avail -k "$primary_tmp" | tail -1) < 58*1024*1024 )); then
    primary_tmp="$primary_root/scratch/$SLURM_JOB_ID"
    mkdir -p "$primary_tmp"
fi
export TMPDIR="$primary_tmp" SLURM_TMPDIR="$primary_tmp"
for leg in 0 1; do
    python -u /home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/measure_primary_replay.py full "$SLURM_ARRAY_TASK_ID" "$leg"
done

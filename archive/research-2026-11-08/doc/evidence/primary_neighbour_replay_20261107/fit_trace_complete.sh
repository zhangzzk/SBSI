#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --nodelist=th-cl-262h05n3
#SBATCH --time=00:30:00
#SBATCH --job-name=primary_fitexact
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107/logs/%x_%j.out
set -eo pipefail
source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu_galsbi_resim
python -u /home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/primary_replay_fit_trace_complete.py
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u /home/z/Zekang.Zhang/SBSI/doc/evidence/primary_neighbour_replay_20261107/fit_trace_report.py merged

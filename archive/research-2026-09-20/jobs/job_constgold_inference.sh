#!/usr/bin/env bash
# Stage resources and dependencies are supplied by the frozen run submission.
# RUN_ROOT contains run_stage.py, source hashes, and an isolated installed venv.
#SBATCH --job-name=sbsi_cg500k
#SBATCH --partition=inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${RUN_ROOT:?set RUN_ROOT to a frozen inference run directory}"
unset PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
exec "$RUN_ROOT/venv/bin/python" -u "$RUN_ROOT/run_stage.py" "$1"

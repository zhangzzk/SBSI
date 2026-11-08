#!/bin/bash
#SBATCH --job-name=sbsi-tests
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/tests_%j.out
set -euo pipefail
# Submit from the checkout root. SBSI_TEST_PYTHON may select another editable installation.
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
# Keep bytecode local to this run; avoid stale shared-filesystem test caches.
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-pycache-${SLURM_JOB_ID:-$$}"
SBSI_TEST_PYTHON=${SBSI_TEST_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
"$SBSI_TEST_PYTHON" -c 'import pathlib, sbsi; assert pathlib.Path(sbsi.__file__).resolve().parent.parent == pathlib.Path.cwd(), sbsi.__file__'
"$SBSI_TEST_PYTHON" -m pytest tests -q

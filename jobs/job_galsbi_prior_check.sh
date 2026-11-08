#!/bin/bash
#SBATCH --job-name=sbsi-galsbi-prior
#SBATCH --partition=inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd "${SLURM_SUBMIT_DIR:?submit from the SBSI checkout}"
GALSBI_CHECK_OUTPUT=${GALSBI_CHECK_OUTPUT:?set an explicit external output directory}
GALSBI_CHECK_PYTHON=${GALSBI_CHECK_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
"$GALSBI_CHECK_PYTHON" -u scripts/check_galsbi_continuous_prior.py "${GALSBI_CHECK_STAGE:-all}" --output "$GALSBI_CHECK_OUTPUT"

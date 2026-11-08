#!/bin/bash
#SBATCH --job-name=sbsi-galsbi-neighbours
#SBATCH --partition=inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd "${SLURM_SUBMIT_DIR:?submit from SBSI checkout}"
GALSBI_CHECK_OUTPUT=${GALSBI_CHECK_OUTPUT:?set output directory}
GALSBI_CHECK_REFERENCE=${GALSBI_CHECK_REFERENCE:?set isolated pilot reference directory}
GALSBI_CHECK_PYTHON=${GALSBI_CHECK_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
"$GALSBI_CHECK_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$GALSBI_CHECK_OUTPUT/resources.json"
"$GALSBI_CHECK_PYTHON" -u scripts/check_galsbi_neighbours.py --output "$GALSBI_CHECK_OUTPUT" --reference-root "$GALSBI_CHECK_REFERENCE"

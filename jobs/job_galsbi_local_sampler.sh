#!/bin/bash
#SBATCH --job-name=sbsi-local-mh
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd "${SLURM_SUBMIT_DIR:?submit from SBSI checkout}"
SBSI_LOCAL_OUTPUT=${SBSI_LOCAL_OUTPUT:?set a fresh output directory}
SBSI_LOCAL_REFERENCE=${SBSI_LOCAL_REFERENCE:?set completed summary pilot directory}
SBSI_LOCAL_PYTHON=${SBSI_LOCAL_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
export SBSI_CACHE_DIR=${SBSI_CACHE_DIR:?set model cache root}
export BLENDEMU_RUNS_DIR=${BLENDEMU_RUNS_DIR:?set response model root}
mkdir -p "$SBSI_LOCAL_OUTPUT"
"$SBSI_LOCAL_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_LOCAL_OUTPUT/resources.json"
"$SBSI_LOCAL_PYTHON" -u scripts/check_galsbi_local_sampler.py --output "$SBSI_LOCAL_OUTPUT" --reference "$SBSI_LOCAL_REFERENCE" "$@"

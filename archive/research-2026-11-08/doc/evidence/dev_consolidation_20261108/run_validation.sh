#!/bin/bash
#SBATCH --job-name=sbsi-merged-tree-tests
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/z/Zekang.Zhang/SBSI/logs/consolidation_%j.out
set -euo pipefail
unset PYTHONPATH SBSI_CACHE_DIR BLENDEMU_RUNS_DIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_TEST_ROOT=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sbsi-consolidation-${SLURM_JOB_ID}.XXXX")
git -C /home/z/Zekang.Zhang/SBSI archive 5e4f601bb01b89941a680c0f70994cc9a41f08ff | tar -x -C "$TASK_TEST_ROOT"
cd "$TASK_TEST_ROOT"
export PYTHONPYCACHEPREFIX="$TASK_TEST_ROOT/.bytecode"
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PY" -c 'import pathlib,sbsi; assert pathlib.Path(sbsi.__file__).resolve().parent.parent == pathlib.Path.cwd(), sbsi.__file__'
"$PY" -m sbsi validate-model like-1.0-galsbi galsbi-multistage
"$PY" -m pytest tests/test_galsbi_model_integration.py tests/test_model_presets.py -q
"$PY" scripts/run_galsbi_inference.py --help > /dev/null

"$PY" -m scripts.run_galsbi_inference --help > /dev/null

"$PY" -m pytest tests -q

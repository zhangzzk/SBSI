#!/bin/bash
#SBATCH --job-name=sbsi-v39-persistent
#SBATCH --partition=cip,inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v39_persistent_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-v39-${SLURM_JOB_ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
SBSI_STUDY_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
: "${SBSI_STUDY_ROOT:?Set the existing pinned study root}"
if [[ "$1" == benchmark ]]; then
  "$SBSI_STUDY_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py -o "$SBSI_STUDY_ROOT/batch_optimization/resources_${SLURM_JOB_ID}.json"
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/batch_optimization/benchmark_v39_batches.py"
elif [[ "$1" == validate ]]; then
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/batch_optimization/validate_v39_batch_effect.py"
elif [[ "$1" == validate_proposal ]]; then
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/batch_optimization/validate_v39_batched_proposal.py"
elif [[ "$1" == validate_proposal24 ]]; then
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/batch_optimization/validate_v39_batched_proposal24.py"
else
  "$SBSI_STUDY_PYTHON" "$SBSI_STUDY_ROOT/batch_optimization/run_v39_persistent.py" "$@"
fi

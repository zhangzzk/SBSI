#!/bin/bash
#SBATCH --job-name=sbsi-signsym-forward
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
root=$1
training=$2
stage=$3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-signsym-forward-${SLURM_JOB_ID}"
cd /tmp
python="$training/venv/bin/python"
"$python" "$root/detect_resources.py" -o "$root/resources_${SLURM_JOB_ID}.json"
"$python" "$training/verify_inputs.py" "$training"
case "$stage" in
  evaluate) "$python" -u "$root/evaluate.py" --root "$root" --chunk "$SLURM_ARRAY_TASK_ID" ;;
  summary) "$python" -u "$root/evaluate.py" --root "$root" --summary ;;
  *) exit 2 ;;
esac

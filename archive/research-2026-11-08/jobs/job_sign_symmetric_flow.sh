#!/bin/bash
#SBATCH --job-name=sbsi-signsym
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
root=$1
stage=${2:-train}
if [[ "$stage" != train ]]; then
  echo "Only train is supported: density fit, base response guard, and symmetry audit." >&2
  exit 2
fi
python="$root/venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-signsym-${SLURM_JOB_ID}"
cd /tmp
"$python" "$root/detect_resources.py" -o "$root/resources_${SLURM_JOB_ID}.json"
"$python" "$root/verify_inputs.py" "$root"
"$python" -m pytest "$root/source_snapshot/tests/test_flow_sign_symmetric.py" -q
"$python" -u "$root/audit_sign_symmetric_flow.py" --root "$root" --output "$root/preflight.json"
"$python" -u "$root/run_stage.py" --root "$root" --stage base

#!/bin/bash
# Prepared only: root must review CPU tests and coordinate dependencies before submission.
#SBATCH --job-name=sbsi-phase-n15
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SBSI_PHASE_SOURCE:?Set the frozen source snapshot}"
: "${SBSI_PHASE_ROOT:?Set sampler_search_20261021 input root}"
: "${SBSI_PHASE_OUTPUT:?Set a fresh diagnostic output directory}"
: "${SBSI_CACHE_DIR:?Set the existing model cache root}"
: "${BLENDEMU_RUNS_DIR:?Set the existing emulator artifact root}"
cd "$SBSI_PHASE_SOURCE"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR:-/tmp}/sbsi-phase-n15-${SLURM_JOB_ID}"
SBSI_PHASE_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$SBSI_PHASE_PYTHON" -c 'import torch; name=torch.cuda.get_device_name(0); assert "V100" in name, name; print("PHASE_DEVICE",name,flush=True)'
for index in 0 1 2 3; do
  test -f "$SBSI_PHASE_ROOT/dev_v1_$index/result.json"
done
for version in v5 v6v100 v7; do
  for index in 0 1 2 3 4 5 6 7; do
    test -f "$SBSI_PHASE_ROOT/dev_${version}_$index/result.json"
  done
done
mkdir "$SBSI_PHASE_OUTPUT"
pilot_inputs=()
for index in 0 1 2 3; do
  output="$SBSI_PHASE_OUTPUT/dev_v1_$index.json"
  "$SBSI_PHASE_PYTHON" -m scripts.check_particle_phase_control \
    --run "$SBSI_PHASE_ROOT/dev_v1_$index" --output "$output" --device cuda
  pilot_inputs+=("$output")
done
"$SBSI_PHASE_PYTHON" -m scripts.report_particle_phase_control \
  --diagnostics "${pilot_inputs[@]}" --output "$SBSI_PHASE_OUTPUT/pilot_v1_controls.json"
coefficients="$SBSI_PHASE_OUTPUT/frozen_v1_coefficients.json"
"$SBSI_PHASE_PYTHON" -m scripts.fit_particle_phase_control \
  --diagnostics "${pilot_inputs[@]}" --output "$coefficients"
for version in v5 v6v100 v7; do
  group_inputs=()
  for index in 0 1 2 3 4 5 6 7; do
    output="$SBSI_PHASE_OUTPUT/dev_${version}_$index.json"
    "$SBSI_PHASE_PYTHON" -m scripts.check_particle_phase_control \
      --run "$SBSI_PHASE_ROOT/dev_${version}_$index" --output "$output" \
      --coefficients "$coefficients" --device cuda
    group_inputs+=("$output")
  done
  "$SBSI_PHASE_PYTHON" -m scripts.report_particle_phase_control \
    --diagnostics "${group_inputs[@]}" --coefficients "$coefficients" \
    --output "$SBSI_PHASE_OUTPUT/${version}_controls.json"
done

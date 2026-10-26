#!/bin/bash
#SBATCH --job-name=sbsi-derivative-reference
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v36_derivative_reference_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
: "${PROBE_JOB_ID:?Set the completed likelihood-probe array ID}"
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_derivative_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests/test_disk_likelihood_failure_probe.py -q
"$PYTHON" scripts/probe_disk_derivative_reference.py --run "$RUN_ROOT" \
  --diagnostics "$RUN_ROOT/likelihood_probe_${PROBE_JOB_ID}" \
  --output "$RUN_ROOT/likelihood_probe_${PROBE_JOB_ID}/derivative_reference_${SLURM_JOB_ID}.json"

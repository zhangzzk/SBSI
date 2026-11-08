#!/bin/bash
#SBATCH --job-name=sbsi-v36-assemble
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v36_assemble_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" scripts/run_disk_inference.py assemble --subset "$RUN_ROOT/prior_subset24m/manifest.json" \
  --input "$RUN_ROOT/input" --prepared "$RUN_ROOT/disk_cache_v1" \
  --output "$RUN_ROOT/disk_assembled_v1" --device cpu

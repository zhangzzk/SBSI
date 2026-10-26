#!/bin/bash
#SBATCH --job-name=sbsi-v36-lru-check
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_lru_check_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_lru_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests -q
"$PYTHON" scripts/run_disk_inference.py run --subset "$RUN_ROOT/prior_subset24m/manifest.json" \
  --input "$RUN_ROOT/input" --prepared "$RUN_ROOT/disk_assembled_v1" \
  --count 256 --object-chunk 16 --compile-flow --output "$RUN_ROOT/fullprior_lru_pilot_${SLURM_JOB_ID}"
"$PYTHON" scripts/check_disk_pilot_replay.py \
  --reference "$RUN_ROOT/fullprior_pilot_16605622" --replay "$RUN_ROOT/fullprior_lru_pilot_${SLURM_JOB_ID}"

#!/bin/bash
#SBATCH --job-name=sbsi-v36-e2e
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_e2e_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
PILOT_ROOT="$RUN_ROOT/pilot_${SLURM_JOB_ID}"
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_e2e_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests -q
COMMON=(--subset "$RUN_ROOT/prior_subset24m/manifest.json" --input "$RUN_ROOT/input")
"$PYTHON" scripts/run_disk_inference.py prepare "${COMMON[@]}" --shard-index 0 --limit-atoms 16384 \
  --output "$PILOT_ROOT/prepared/shard_00"
"$PYTHON" scripts/run_disk_inference.py assemble "${COMMON[@]}" --pilot \
  --prepared "$PILOT_ROOT/prepared" --output "$PILOT_ROOT/assembled"
"$PYTHON" scripts/run_disk_inference.py run "${COMMON[@]}" --pilot --count 128 \
  --prepared "$PILOT_ROOT/assembled" --output "$PILOT_ROOT/inference"

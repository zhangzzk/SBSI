#!/bin/bash
#SBATCH --job-name=sbsi-v36-cache
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/v36_cache_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
printf -v SHARD '%02d' "$SLURM_ARRAY_TASK_ID"
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_cache_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"$PYTHON" scripts/run_disk_inference.py prepare --subset "$RUN_ROOT/prior_subset24m/manifest.json" \
  --input "$RUN_ROOT/input" --shard-index "$SLURM_ARRAY_TASK_ID" \
  --output "$RUN_ROOT/disk_cache_v1/shard_$SHARD"

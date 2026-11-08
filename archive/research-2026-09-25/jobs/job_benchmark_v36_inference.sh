#!/bin/bash
#SBATCH --job-name=sbsi-v36-benchmark
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=logs/v36_benchmark_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PYTHON" -m pytest tests/test_catalogue_disk_response.py -q
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py -o "logs/v36_benchmark_resources_${SLURM_JOB_ID}.json"
"$PYTHON" scripts/benchmark_v36_inference.py \
  --parent "$SBSI_CACHE_DIR/fresh_fullscene_joint_20260918_v1/full/parent/case20010.npz" \
  --pairs "$SBSI_CACHE_DIR/fresh_fullscene_joint_20260918_v1/full/pairs/case20010.feather" \
  --output "logs/v36_benchmark_${SLURM_JOB_ID}.json"

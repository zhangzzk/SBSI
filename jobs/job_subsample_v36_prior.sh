#!/bin/bash
#SBATCH --job-name=sbsi-v36-24m
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_subset24m_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_subset_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests -q
"$PYTHON" scripts/subsample_disk_prior.py --source "$RUN_ROOT/prior" \
  --output "$RUN_ROOT/prior_subset24m" --size 24000000 --seed 20260920

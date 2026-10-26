#!/bin/bash
#SBATCH --job-name=sbsi-disk-review
#SBATCH --partition=inter,cip
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_population_review_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_population_resources_${SLURM_JOB_ID}.json"
"$PYTHON" -m pytest tests -q
"$PYTHON" scripts/audit_disk_population.py --run "$RUN_ROOT" \
  --output "$RUN_ROOT/population_review_${SLURM_JOB_ID}"

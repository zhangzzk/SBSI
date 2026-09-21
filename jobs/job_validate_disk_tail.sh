#!/bin/bash
#SBATCH --job-name=sbsi-tail-exact
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v36_tail_exact_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_tail_exact_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"$PYTHON" -m pytest tests/test_disk_curvature_probe.py tests/test_disk_auxiliary_probe.py -q
"$PYTHON" scripts/validate_disk_tail_integration.py --run "$RUN_ROOT" --worker "$SLURM_ARRAY_TASK_ID" \
  --output "$RUN_ROOT/tail_exact_${SLURM_ARRAY_JOB_ID}/worker_${SLURM_ARRAY_TASK_ID}"

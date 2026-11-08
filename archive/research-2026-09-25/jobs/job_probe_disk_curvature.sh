#!/bin/bash
#SBATCH --job-name=sbsi-curvature-probe
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v36_curvature_probe_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
MODE=exact
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then MODE=sample; fi
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "logs/v36_curvature_probe_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"$PYTHON" scripts/probe_disk_curvature.py --run "$RUN_ROOT" --mode "$MODE" \
  --output "$RUN_ROOT/curvature_probe_${SLURM_ARRAY_JOB_ID}_${MODE}"

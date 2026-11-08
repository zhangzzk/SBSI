#!/bin/bash
#SBATCH --job-name=sbsi-newmodels-full
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/v36_newmodels_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
ROOT="$SBSI_CACHE_DIR/v36_domain_recovery_20260924_v1"
NAMES=(unbounded closecrowd closecrowd_emulator265)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
"$PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "logs/v36_newmodel_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
"$PYTHON" -m pytest tests/test_v36_domain_response.py tests/test_v36_saved_models.py -q
"$PYTHON" -u scripts/audit_v36_domain_response.py \
  --catalogues /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --parent "$SBSI_CACHE_DIR/constgold_truth_parent_20260917_v1/parent_v2" \
  --reference "$SBSI_CACHE_DIR/scene_classifier_geometry_20260918_v1/smooth_joint_main.json" \
  --candidate "$ROOT/candidates/$NAME.json" \
  --candidate-parent "$SBSI_CACHE_DIR/constgold_truth_parent_closecrowd_20260924_v1/parent" \
  --candidate-pairs "$SBSI_CACHE_DIR/constgold_truth_parent_unbounded_20260922_v1/pairs" \
  --pair-query-k 20 \
  --output "$ROOT/new_${NAME}_${SLURM_ARRAY_JOB_ID}" --cases 40 41

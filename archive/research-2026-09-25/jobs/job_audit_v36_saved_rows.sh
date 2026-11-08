#!/bin/bash
#SBATCH --job-name=sbsi-model-rows
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/v36_saved_rows_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
NAMES=(unbounded_rows closecrowd closecrowd_guard_ep03)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
"$PYTHON" -m pytest tests/test_v36_saved_models.py -q
"$PYTHON" -u scripts/audit_v36_saved_rows.py --model "$NAME" --workers 4 \
 --output "/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/rows_${NAME}.json"

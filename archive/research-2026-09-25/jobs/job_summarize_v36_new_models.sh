#!/bin/bash
#SBATCH --job-name=sbsi-newmodel-summary
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/v36_newmodel_summary_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
ROOT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1
: "${AUDIT_RUN:?Missing audit array job id}"
for NAME in ${AUDIT_NAMES:-unbounded closecrowd}; do
 "$PYTHON" scripts/summarize_v36_domain_response.py \
   --roots "$ROOT/new_${NAME}_${AUDIT_RUN}" \
   --output "$ROOT/new_${NAME}_${AUDIT_RUN}_summary.json"
done

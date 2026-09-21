#!/usr/bin/env bash
# CPU preparation/review or GPU training for the minimal R_blend flow test.

#SBATCH --job-name=sbsi-mrb-flow
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/logs/%x_%A_%a.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python
export PYTHONPATH="$source_root"
cd /tmp

case "${PHASE:?Set PHASE=prepare,train,review}" in
  prepare|review)
    "$python" -u /home/z/Zekang.Zhang/SBSI/scripts/train_minimal_rblend_flow_condition.py \
      "$PHASE" --root "$root"
    ;;
  train)
    : "${SLURM_ARRAY_TASK_ID:?train requires array indices 0-1}"
    "$python" -u /home/z/Zekang.Zhang/SBSI/scripts/train_minimal_rblend_flow_condition.py \
      train --root "$root" --index "$SLURM_ARRAY_TASK_ID"
    ;;
  *)
    echo "unknown PHASE=$PHASE" >&2
    exit 2
    ;;
esac

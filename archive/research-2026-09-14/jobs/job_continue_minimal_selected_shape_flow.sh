#!/usr/bin/env bash
# NLL-only continuation or review of the minimal selected-shape flow.

#SBATCH --job-name=sbsi-ms-cont
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_continuation_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_continuation_v1/logs/%x_%j.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_continuation_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

cd /home/z/Zekang.Zhang/SBSI
"$python" -u -m scripts.continue_minimal_selected_shape_flow \
  "${PHASE:?Set PHASE=train,review}" --root "$root"

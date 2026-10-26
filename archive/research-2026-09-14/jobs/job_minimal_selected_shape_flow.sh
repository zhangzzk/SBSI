#!/usr/bin/env bash
# Existing-catalogue preparation, GPU training, or review for the selected-shape flow.

#SBATCH --job-name=sbsi-ms-shape
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_v1/logs/%x_%j.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_flow_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

cd /home/z/Zekang.Zhang/SBSI
"$python" -u -m scripts.train_minimal_selected_shape_flow \
  "${PHASE:?Set PHASE=prepare,train,review}" --root "$root"

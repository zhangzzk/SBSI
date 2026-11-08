#!/bin/bash
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=sbsi-selected-review
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2/logs/%j_review.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2/logs/%j_review.err
set -euo pipefail

unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source

cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python -u \
  /home/z/Zekang.Zhang/SBSI/scripts/train_minimal_selected_event_head.py review \
  --output-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2

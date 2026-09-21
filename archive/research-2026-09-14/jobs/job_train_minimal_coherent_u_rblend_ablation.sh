#!/bin/bash
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%4
#SBATCH --job-name=minimal-rblend
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/logs/%A_%a_train.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/logs/%A_%a_train.err
set -euo pipefail

unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source

cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python -u \
  /home/z/Zekang.Zhang/SBSI/scripts/train_minimal_coherent_u_rblend_ablation.py train \
  --output-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2 \
  --index "$SLURM_ARRAY_TASK_ID"

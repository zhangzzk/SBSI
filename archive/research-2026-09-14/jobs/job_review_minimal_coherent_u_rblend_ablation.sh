#!/bin/bash
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=review-min-rblend
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/logs/%j_review.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/logs/%j_review.err
set -euo pipefail

unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source

cd /tmp
/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/env/bin/python -u \
  /home/z/Zekang.Zhang/SBSI/scripts/train_minimal_coherent_u_rblend_ablation.py review \
  --output-root /project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2

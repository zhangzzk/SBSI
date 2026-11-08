#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fg0finalcmp
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
v2="$cache/fixed_g0_m258_r060_v2"
historical="$cache/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/paired/epoch154.pt"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -u -m scripts.compare_fixed_g0_models \
  --domain-root "$v2/domain" \
  --nll-indices "$v2/flow/validation_nll_indices.npy" \
  --response-indices "$v2/flow/validation_response_pair_indices.npy" \
  --model "historical_lambda10_epoch154=$historical" \
  --model "fixed_v1_nll=$cache/fixed_g0_m258_r060_v1/flow/nll/selected.pt" \
  --model "fixed_v1_staged=$cache/fixed_g0_m258_r060_v1/flow/paired/selected.pt" \
  --model "fixed_v1_direct=$cache/fixed_g0_m258_r060_v1/flow_direct_response/selected.pt" \
  --model "fixed_v2_nll=$v2/flow/nll/selected.pt" \
  --model "fixed_v2_staged=$v2/flow/paired/selected.pt" \
  --model "fixed_v2_direct=$v2/flow_direct_response/selected.pt" \
  --response-draws 64 --response-groups 4 \
  --output "$v2/performance_review_20260914/final_common_v2_scores.json"

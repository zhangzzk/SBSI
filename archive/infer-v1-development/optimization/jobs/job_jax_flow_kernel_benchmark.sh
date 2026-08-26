#!/usr/bin/env bash
# Archived Infer V1 optimization job.
# Compare identical frozen-flow rows under Torch and JAX on one GPU.

#SBATCH --job-name=sbsi_jax_flow
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

: "${OUTPUT:=$root/jax_flow_kernel_n65536_v9_v1}"
: "${FLOW:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt}"
: "${ROWS_PER_VIEW:=65536}"
: "${VIEWS:=9}"
: "${REPEATS:=7}"
: "${JAX_OVERLAY:=/project/ls-gruen/users/zekang.zhang/sbsi/env_overlays/jax_cuda12_0p6p2}"
: "${FLOW_CONTEXT_PARQUET:=$root/model_cache_section5_v1/flow_zero.parquet}"

if [[ ! -d "$JAX_OVERLAY/jax_plugins" ]]; then
  echo "JAX CUDA overlay is missing: $JAX_OVERLAY" >&2
  exit 2
fi
export PYTHONPATH="$JAX_OVERLAY:$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

optional_args=()
if [[ -n "$FLOW_CONTEXT_PARQUET" ]]; then
  optional_args+=(--flow-context-parquet "$FLOW_CONTEXT_PARQUET")
fi

"$python" "$repo/scripts/benchmark_jax_flow_kernel.py" \
  --measurement-model "$FLOW" --output "$OUTPUT" \
  --rows-per-view "$ROWS_PER_VIEW" --views "$VIEWS" \
  --repeats "$REPEATS" --device cuda "${optional_args[@]}"

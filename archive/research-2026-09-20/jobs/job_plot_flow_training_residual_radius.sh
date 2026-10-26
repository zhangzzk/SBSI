#!/usr/bin/env bash
# Plot staged-flow response residual versus fixed-g0 radius on cached train cases.

#SBATCH --job-name=sbsi_frtrain
#SBATCH --partition=cip,inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
predictions=$v2/flow_self_response_v1/R_self_model_trained.feather
prediction_manifest=$v2/flow_self_response_v1/R_self_model_trained.manifest.json
output_prefix=$repo/plots/figures/flow_training_residual_vs_flux_radius_c20_71

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" -u plots/flow_training_residual_vs_flux_radius.py \
  --domain-root "$v2/domain" \
  --predictions "$predictions" \
  --prediction-manifest "$prediction_manifest" \
  --output-prefix "$output_prefix"

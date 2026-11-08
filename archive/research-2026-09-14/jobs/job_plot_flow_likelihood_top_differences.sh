#!/usr/bin/env bash

#SBATCH --job-name=sbsi_flow_topdiff
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_likelihood_topdiff_v32_v33_cg100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_likelihood_topdiff_v32_v33_cg100k_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_likelihood_topdiff_v32_v33_cg100k_v1
output=$root/flow_likelihood_top64_v32_v33_constgold100k_v3
v32=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
v33=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt

mkdir -p "$root/logs"
echo "38a76bb9bbece61f403ce2781883f38779b419101d0e690439edffb82b93c51e  $v32" | sha256sum --check --status
echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $v33" | sha256sum --check --status

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
"$python" -u "$repo/scripts/plot_flow_likelihood_top_differences.py" \
  --source-sample /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  --v32-model "$v32" --v33-model "$v33" \
  --output "$output" --screen-draws 512 --screen-directions 8 \
  --shortlist 2048 --final-draws 8192 --final-directions 32 --show 64 \
  --trim-fraction 0.005 --context-workers 8 --seed 7301 --shear 0.02 --device cuda

echo "FLOW_LIKELIHOOD_TOPDIFF_DONE output=$output.png"

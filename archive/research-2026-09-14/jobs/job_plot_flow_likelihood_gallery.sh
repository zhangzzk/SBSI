#!/usr/bin/env bash

#SBATCH --job-name=sbsi_flow_gallery
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_likelihood_gallery_v32_v33_cg36_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_likelihood_gallery_v32_v33_cg36_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_likelihood_gallery_v32_v33_cg36_v1
output=$root/flow_likelihood_gallery_v32_v33_constgold36_v2
v32=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
v33=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt

mkdir -p "$root/logs"
echo "38a76bb9bbece61f403ce2781883f38779b419101d0e690439edffb82b93c51e  $v32" | sha256sum --check --status
echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $v33" | sha256sum --check --status

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$python" -u "$repo/scripts/plot_flow_likelihood_gallery.py" \
  --source-sample /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  --v32-model "$v32" --v33-model "$v33" \
  --output "$output" --draws 8192 --seed 7301 --shear 0.02 \
  --extent 1.0 --device cuda

echo "FLOW_LIKELIHOOD_GALLERY_DONE output=$output.png"

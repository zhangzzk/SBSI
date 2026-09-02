#!/usr/bin/env bash
# Twenty-case response ladder with the 500/500 Flow E and V3.2 transition detector.

#SBATCH --job-name=sbsi_cg_Rlad_E500tr
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_E500_transition_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_E500_transition_v2/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
constroot=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_E500_transition_v2
output=$root/result.json
flow=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
[[ -s "$flow" ]] || { echo "missing measurement model $flow" >&2; exit 2; }
[[ -s "$classifier" ]] || { echo "missing detection classifier $classifier" >&2; exit 2; }
echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $flow" | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" | sha256sum --check --status

cases=()
for case in $(seq 40 59); do
  cases+=(--case "$case")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root "$constroot" \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  --rblend-lookup "$cache/mixed_shear_cde/constgold/rblend_v3_c40-49.feather" \
  --rblend-lookup "$cache/mixed_shear_cde/constgold/rblend_v3_c50-59.feather" \
  --measurement-model "$flow" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact \
  --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda

echo "CONSTGOLD_RESPONSE_LADDER20_E500_TRANSITION_DONE output=$output"

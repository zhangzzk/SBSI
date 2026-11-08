#!/usr/bin/env bash
# V3.3-like three-experiment ConstGold response validation, cases 40--139.

#SBATCH --job-name=sbsi_v33_cg100
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v33_constgold_response_experiments_c40_139_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v33_constgold_response_experiments_c40_139_v1/logs/%x_%j.err

set -euo pipefail

repository=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
constgold_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
output_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v33_constgold_response_experiments_c40_139_v1
output=$output_root/result.json
flow=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

mkdir -p "$output_root/logs"
[[ ! -e "$output" ]] || {
  echo "refusing to overwrite $output" >&2
  exit 2
}
[[ -s "$flow" ]] || { echo "missing measurement model $flow" >&2; exit 2; }
[[ -s "$classifier" ]] || {
  echo "missing detection classifier $classifier" >&2
  exit 2
}
echo "4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73  $flow" \
  | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" \
  | sha256sum --check --status

cases=()
for case in $(seq 40 139); do
  cases+=(--case "$case")
done

rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  path=$cache/mixed_shear_cde/constgold/rblend_v3_c${first}-${last}.feather
  [[ -s "$path" ]] || { echo "missing R_blend lookup $path" >&2; exit 2; }
  rblend+=(--rblend-lookup "$path")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repository${PYTHONPATH:+:$PYTHONPATH}"
cd "$repository"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root "$constgold_root" \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  "${rblend[@]}" \
  --measurement-model "$flow" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact \
  --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 \
  --draws 64 \
  --sampling-seed 7301 \
  --batch-size 2048 \
  --n-boot 10000 \
  --bootstrap-seed 20260902 \
  --device cuda

echo "V33_CONSTGOLD_RESPONSE_EXPERIMENTS_C40_139_DONE output=$output"

#!/usr/bin/env bash
# Twenty-case ConstGold response audit for the density-admissible ladder choice.

#SBATCH --job-name=sbsi_Etheta_R20
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/constgold_response_c40_59/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/constgold_response_c40_59/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
constroot=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
screen=$cache/mixed_shear_e_coupling_ladder_v1
root=$screen/constgold_response_c40_59
output=$root/result.json
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
[[ -s "$screen/summary.json" ]] || { echo "missing ladder summary" >&2; exit 2; }
readarray -t selected < <("$python" - "$screen/summary.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))["selected"]
print(row["model"])
print(row["model_sha256"])
PY
)
flow=${selected[0]}
flow_sha=${selected[1]}
echo "$flow_sha  $flow" | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" \
  | sha256sum --check --status

cases=()
for case in $(seq 40 59); do cases+=(--case "$case"); done

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
  --detection-neighbour-selection impact --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda

echo "CONSTGOLD_SELECTED_FLOW_RESPONSE20_DONE output=$output"

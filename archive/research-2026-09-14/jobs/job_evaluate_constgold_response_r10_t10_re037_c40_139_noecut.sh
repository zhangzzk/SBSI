#!/usr/bin/env bash
# All-100-case Re>0.37 ConstGold closure with the measured |e| cut disabled.

#SBATCH --job-name=sbsi_cg100_r037_noe
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/response_c40_139_noecut_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/response_c40_139_noecut_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
domain_audit=$cache/plain_complete_flow_response_re037_v1/constgold/response_c40_139_v1/domain_audit.json
root=$cache/plain_complete_flow_response_re037_v1/constgold/response_c40_139_noecut_v1
output=$root/result.json
flow=$cache/plain_complete_flow_response_re037_v1/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

mkdir -p "$root/logs"
[[ -s "$domain_audit" ]] || { echo "missing passed domain audit" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
echo "0ca0a9847e17dfb0e15dc1c13df5a633ff3342c45f3e8bf53fb4b3a40c99700b  $flow" | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" | sha256sum --check --status
echo "f1ad4ef270c976588f904a1be081a0dbc254926d72f24b9d29785c93627082fd  $domain_audit" | sha256sum --check --status

cases=()
for case in $(seq 40 139); do cases+=(--case "$case"); done
rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  rblend+=(--rblend-lookup "$cache/plain_complete_flow_response_re037_v1/constgold/rblend_re037_complete_c${first}-${last}.feather")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  "${rblend[@]}" \
  --measurement-model "$flow" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda \
  --disable-shape-cut

echo "CONSTGOLD_R10_T10_RE037_RESPONSE100_NOECUT_DONE output=$output"

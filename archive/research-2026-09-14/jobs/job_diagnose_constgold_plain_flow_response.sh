#!/usr/bin/env bash
# Diagnose sampling, checkpoint, and measured-cut sources of the Re>0.37 closure gap.

#SBATCH --job-name=sbsi_cgdiag_r037
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/diagnostics_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/diagnostics_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
domain_audit=$base/constgold/response_c40_139_v1/domain_audit.json
root=$base/constgold/diagnostics_v1
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt
swa=$base/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt
single=$base/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swabase.pt
pure=$base/measurement_flow_plain_pure_nll_re037_s501_swaavg.pt

mkdir -p "$root/logs"
[[ -s "$domain_audit" ]] || { echo "missing passed domain audit" >&2; exit 2; }
echo "f1ad4ef270c976588f904a1be081a0dbc254926d72f24b9d29785c93627082fd  $domain_audit" | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" | sha256sum --check --status

task=${SLURM_ARRAY_TASK_ID:?array task is required}
model=$swa
model_hash=0ca0a9847e17dfb0e15dc1c13df5a633ff3342c45f3e8bf53fb4b3a40c99700b
draws=64
seed=7301
tag=
flags=()
case "$task" in
  0) tag=swa_d64_seed7302_all; seed=7302 ;;
  1) tag=swa_d256_seed7301_all; draws=256 ;;
  2) tag=swa_d64_seed7302_noe; seed=7302; flags+=(--disable-shape-cut) ;;
  3) tag=swa_d256_seed7301_noe; draws=256; flags+=(--disable-shape-cut) ;;
  4) tag=swa_d64_seed7301_radius_only; flags+=(--disable-shape-cut --disable-mag-cut) ;;
  5) tag=swa_d64_seed7301_mag_only; flags+=(--disable-shape-cut --disable-radius-cut) ;;
  6) tag=swa_d64_seed7301_nocuts; flags+=(--disable-shape-cut --disable-mag-cut --disable-radius-cut) ;;
  7)
    tag=single_d64_seed7301_all
    model=$single
    model_hash=f58a1980124e51d47d7858bb3a83ccff64bffc81303dd11f28c70ce2e3442940
    ;;
  8)
    tag=purenll_d64_seed7301_all
    model=$pure
    model_hash=d0ac7b040719710df5b9a262715a49e3678dec35340029864e22178dd28ee114
    ;;
  *) echo "unknown task $task" >&2; exit 2 ;;
esac
echo "$model_hash  $model" | sha256sum --check --status

output=$root/$tag.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
cases=()
for case_id in $(seq 40 139); do cases+=(--case "$case_id"); done
rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  rblend+=(--rblend-lookup "$base/constgold/rblend_re037_complete_c${first}-${last}.feather")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  "${rblend[@]}" \
  --measurement-model "$model" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws "$draws" --sampling-seed "$seed" --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda \
  "${flags[@]}"

echo "CONSTGOLD_PLAIN_FLOW_DIAGNOSTIC_DONE task=$task tag=$tag output=$output"

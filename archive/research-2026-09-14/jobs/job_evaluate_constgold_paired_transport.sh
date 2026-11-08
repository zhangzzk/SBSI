#!/usr/bin/env bash
# Exact 100-case ConstGold cut decomposition for the Re>0.37 100/100 flow.

#SBATCH --job-name=sbsi_cg_pair1
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_train_v1/constgold/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_train_v1/constgold/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/paired_transport_train_v1/constgold
flow=$base/paired_transport_train_v1/measurement_flow_paired1_r100_t100_full_re037_s501_swabase.pt
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt
domain_audit=$root/domain_audit.json

mkdir -p "$root/logs"
[[ -s "$domain_audit" ]] || { echo "missing domain audit" >&2; exit 2; }
"/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python" - "$domain_audit" "$flow" <<'PY'
from hashlib import sha256
import json
import sys

audit_path, flow_path = sys.argv[1:]
audit = json.load(open(audit_path))
h = sha256(open(flow_path, "rb").read()).hexdigest()
if audit.get("status") != "pass" or audit.get("case_count") != 100:
    raise SystemExit("domain audit did not pass all 100 cases")
if audit.get("primary_domain") != {"r_input": [18.0, 25.8], "Re_input": [0.37, 1.5]}:
    raise SystemExit("domain audit has the wrong primary support")
if audit.get("flow", {}).get("sha256") != h:
    raise SystemExit("domain audit and flow hashes differ")
PY
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" | sha256sum --check --status

task=${SLURM_ARRAY_TASK_ID:?array task is required}
flags=()
case "$task" in
  0) tag=all ;;
  1) tag=no_shape; flags+=(--disable-shape-cut) ;;
  2) tag=radius_only; flags+=(--disable-shape-cut --disable-mag-cut) ;;
  3) tag=mag_only; flags+=(--disable-shape-cut --disable-radius-cut) ;;
  4) tag=no_cuts; flags+=(--disable-shape-cut --disable-mag-cut --disable-radius-cut) ;;
  *) echo "unknown task $task" >&2; exit 2 ;;
esac
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
  --measurement-model "$flow" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda \
  "${flags[@]}"

sha256sum "$output"
echo "CONSTGOLD_R100_T100_RE037_CUT_DONE task=$task tag=$tag output=$output"

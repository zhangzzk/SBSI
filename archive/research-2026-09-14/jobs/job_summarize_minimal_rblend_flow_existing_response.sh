#!/usr/bin/env bash
# Reduce the reused response screen after its GPU job succeeds.

#SBATCH --job-name=sbsi_mrflow_reduce
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/response_screen/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/response_screen/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
response=$root/response_screen/result.json
development=$root/development_result.json
baseline=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_existing_screen_v1/results.json
output=$root/response_screen/decision.json

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

check_sha 9de50f2bb0a053a9a652cfc1a930c823b6f867fe351c7c4f79f47f369edc6571 "$repo/scripts/summarize_minimal_rblend_flow_existing_response.py"
check_sha f726af5d6beed523b48b31ae53675190a678024a51dbed298e5891d3a0dc5a73 "$development"
check_sha 4d8ad29c73468faacd002ad017c5b80b0607f08076635a5f69e45be65cf1adc2 "$baseline"
[[ -s "$response" && ! -L "$response" ]] || { echo "missing response: $response" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.summarize_minimal_rblend_flow_existing_response \
  --response "$response" \
  --development "$development" \
  --detector-only-baseline "$baseline" \
  --output "$output"

sha256sum "$response" "$output"
echo "MINIMAL_RBLEND_FLOW_EXISTING_REDUCTION_DONE output=$output"

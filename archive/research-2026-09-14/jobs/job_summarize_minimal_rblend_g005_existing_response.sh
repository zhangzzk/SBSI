#!/usr/bin/env bash
# Reduce the existing cases140--239 response under the frozen equivalence gate.

#SBATCH --job-name=sbsi_mrflow_g005_reduce
#SBATCH --partition=inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
experiment=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
root=$experiment/existing_g005_c140_239
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
prior=$experiment/response_screen/decision.json
response=$root/response/result.json
output=$root/response/decision.json

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

[[ -s "$prior" && ! -L "$prior" ]] || { echo "missing prior decision: $prior" >&2; exit 2; }
if ! "$python" - "$prior" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(
    Path(sys.argv[1]).read_text(),
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant {value}")
    ),
)
if payload.get("passes_reused_structural_screen") is not True:
    raise SystemExit(1)
PY
then
  echo "MINIMAL_RBLEND_G005_REDUCTION_SKIPPED structural_gate=false"
  exit 0
fi

check_sha 6b1f749e56b5e2b7d57f3c270d5bf31788d6bf9fb6ce86007bbdaeabb81b22f2 "$repo/scripts/summarize_minimal_rblend_g005_existing_response.py"
[[ -s "$response" && ! -L "$response" ]] || { echo "missing response: $response" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.summarize_minimal_rblend_g005_existing_response \
  --response "$response" \
  --prior-decision "$prior" \
  --output "$output"

sha256sum "$response" "$prior" "$output"
echo "MINIMAL_RBLEND_G005_REDUCTION_DONE output=$output"

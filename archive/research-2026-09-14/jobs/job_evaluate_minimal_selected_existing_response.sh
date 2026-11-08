#!/usr/bin/env bash
# Evaluate the frozen direct-selected candidate on existing cases140--239.

#SBATCH --job-name=sbsi_sel_g005
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%j_response.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%j_response.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239
crowd_lookup=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_g005_c140-239/crowd_flux_conc_c140-239.feather
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/training/flow8_rblend/epoch016.pt
selected=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2/models
output=$root/response/result.json

draws=$(PYTHONPATH="$repo" "$analysis_python" - "$root" <<'PY'
from scripts.summarize_minimal_selected_existing_response import load_protocol
import sys
print(load_protocol(sys.argv[1])["sampling"]["draws_per_row_per_qmc_seed"])
PY
)
[[ "$draws" =~ ^(64|256|1024)$ ]] || { echo "invalid frozen draw budget: $draws" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$analysis_python" - "$root" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()

protocol_hash = digest(root / "protocol.json")
covered = []
for first in range(140, 240, 10):
    last = first + 9
    receipt_path = root / "lookup" / f"rblend_trial9_c{first}-{last}.json"
    receipt = json.loads(receipt_path.read_text())
    cases = list(range(first, last + 1))
    if not receipt.get("checks_passed") or receipt.get("cases") != cases or not receipt.get("trial9_inputs_equal_between_legs"):
        raise RuntimeError(f"invalid lookup receipt: {receipt_path}")
    protocol = str(root / "protocol.json")
    if receipt["artifacts"].get(protocol) != protocol_hash:
        raise RuntimeError(f"lookup protocol mismatch: {receipt_path}")
    for rendered, expected in {**receipt["artifacts"], **receipt["input_hashes"]}.items():
        path = Path(rendered)
        if digest(path) != expected:
            raise RuntimeError(f"lookup artifact changed: {path}")
    covered.extend(cases)
if covered != list(range(140, 240)):
    raise RuntimeError("lookup coverage changed")
print("MINIMAL_SELECTED_LOOKUPS_VERIFIED cases=140-239", flush=True)
PY

cases=()
for case in $(seq 140 239); do cases+=(--case "$case"); done
lookups=()
for first in $(seq 140 10 230); do
  last=$((first + 9))
  lookups+=(--rblend-lookup "$root/lookup/rblend_trial9_c${first}-${last}.feather")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$source_root:$repo"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /tmp
"$python" -u "$repo/scripts/evaluate_constgold_complete_forward_response.py" \
  --constgold-root "$catalogue_root" \
  "${lookups[@]}" \
  --crowd-lookup "$crowd_lookup" \
  --measurement-model "$flow" \
  --detection-classifier "$selected/seed20260913/selected.pt" \
  --detection-classifier "$selected/seed20260914/selected.pt" \
  --detection-classifier "$selected/seed20260915/selected.pt" \
  --probability-factorization direct_selected \
  --detection-neighbour-selection nearest --detection-impact-exponent 2 \
  --shear-component g1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.05 --draws "$draws" \
  --sampling-seed 7301 --sampling-seed 7302 \
  --sampling-seed 7303 --sampling-seed 7304 \
  --batch-size 2048 --device cuda \
  --n-boot 10000 --bootstrap-seed 20260914 \
  --disable-shape-cut

sha256sum "$output"
echo "MINIMAL_SELECTED_EXISTING_RESPONSE_DONE output=$output"

#!/usr/bin/env bash
# Build trial9 R_blend lookups for the frozen existing cases140--239 endpoint.

#SBATCH --job-name=sbsi_sel_rb
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%A_%a_lookup.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1/logs/%A_%a_lookup.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_existing_response_v1
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
blendemu_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/anchorblend_current_re037_v1/blendemu_snapshot
python=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/anchorblend_current_re037_v1/env/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/training/flow8_rblend/epoch016.pt
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_trial9_v1/trial9_base.json
metadata=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_trial9_v1/metadata.json

PYTHONPATH="$repo" "$analysis_python" - "$root" <<'PY'
from scripts.summarize_minimal_selected_existing_response import load_protocol
import sys
load_protocol(sys.argv[1])
print("MINIMAL_SELECTED_RESPONSE_PROTOCOL_VERIFIED", flush=True)
PY

first=$((140 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
output=$root/lookup/rblend_trial9_c${first}-${last}.feather
receipt=$root/lookup/rblend_trial9_c${first}-${last}.json
[[ ! -e "$output" && ! -e "$receipt" ]] || {
  echo "refusing to overwrite $output or $receipt" >&2
  exit 2
}
mapfile -t cases < <(seq "$first" "$last")

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export XGB_DEVICE=cpu
export PYTHONPATH="$source_root:$blendemu_root"
cd /tmp
"$python" -u "$repo/scripts/build_constgold_blend_lookup.py" \
  --cases "${cases[@]}" \
  --input-pattern "$catalogue_root/case{case}_0.05/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" \
  --emulator-model "$emulator" \
  --emulator-sha256 9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d \
  --output "$output"

"$python" - "$first" "$last" "$catalogue_root" "$output" "$receipt" \
  "$root/protocol.json" "$emulator" "$metadata" "$repo/scripts/build_constgold_blend_lookup.py" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.feather as pf

first, last = map(int, sys.argv[1:3])
catalogue_root, output, receipt, protocol, emulator, metadata, builder = map(
    Path, sys.argv[3:]
)

def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()

def input_path(case, sign):
    return catalogue_root / f"case{case}_{sign}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather"

columns = ["index_input", "RA_input", "DEC_input", "Re_input", "sersic_n_input", "r_input"]
result = pf.read_table(output).to_pandas()
if set(result) != {"case", "input_index", "R_blend"} or result.duplicated(["case", "input_index"]).any():
    raise RuntimeError("invalid lookup schema or duplicate keys")
if set(result["case"].unique()) != set(range(first, last + 1)) or not np.isfinite(result["R_blend"]).all():
    raise RuntimeError("invalid lookup case coverage or values")

input_hashes = {}
rows_by_case = {}
for case in range(first, last + 1):
    plus_path = input_path(case, "0.05")
    minus_path = input_path(case, "-0.05")
    plus = pf.read_table(plus_path, columns=columns).to_pandas()
    minus = pf.read_table(minus_path, columns=columns).to_pandas()
    for name in columns:
        if not np.array_equal(plus[name].to_numpy(), minus[name].to_numpy(), equal_nan=True):
            raise RuntimeError(f"truth legs differ: case={case} column={name}")
    eligible = (
        np.isfinite(plus[["r_input", "Re_input"]]).all(axis=1)
        & (plus["r_input"] > 18.0) & (plus["r_input"] < 25.8)
        & (plus["Re_input"] > 0.37) & (plus["Re_input"] < 1.5)
    )
    expected = np.sort(plus.loc[eligible, "index_input"].to_numpy(np.int64))
    actual = np.sort(result.loc[result["case"] == case, "input_index"].to_numpy(np.int64))
    if not np.array_equal(actual, expected):
        raise RuntimeError(f"lookup support mismatch: case={case}")
    rows_by_case[str(case)] = len(actual)
    for truth_path in (plus_path, minus_path):
        catalogue_dir = truth_path.parent.parent
        measured_paths = (
            catalogue_dir / "CrossMatch" / "tile180.0_-0.5_rot0_matched.feather",
            catalogue_dir / "Shapes" / "shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
        )
        for path in (truth_path, *measured_paths):
            input_hashes[str(path)] = digest(path)

payload = {
    "schema_version": 1,
    "analysis": "trial9 lookup for frozen direct-selected existing response",
    "cases": list(range(first, last + 1)),
    "truth_legs": [-0.05, 0.05],
    "trial9_inputs_equal_between_legs": True,
    "rows": len(result),
    "rows_by_case": rows_by_case,
    "artifacts": {
        str(output): digest(output), str(protocol): digest(protocol),
        str(emulator): digest(emulator), str(metadata): digest(metadata),
        str(builder): digest(builder),
    },
    "input_hashes": input_hashes,
    "checks_passed": True,
}
receipt.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(f"MINIMAL_SELECTED_LOOKUP_READY cases={first}-{last} rows={len(result)}", flush=True)
PY

sha256sum "$output" "$receipt"

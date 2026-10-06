#!/usr/bin/env bash
# Build the frozen trial9 R_blend lookup for existing g1=+/-0.05 cases.
# The array is conditional on the response-blind cases120--139 structural gate.

#SBATCH --job-name=sbsi_mrflow_g005_rb
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
experiment=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
root=$experiment/existing_g005_c140_239
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239
decision=$experiment/response_screen/decision.json
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
blendemu_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/anchorblend_current_re037_v1/blendemu_snapshot
python=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/anchorblend_current_re037_v1/env/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=$experiment/training/flow8_rblend/epoch016.pt
emulator_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_trial9_v1
emulator=$emulator_root/trial9_base.json
metadata=$emulator_root/metadata.json

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

check_sha 19bd81844361de83a439c0da3d743a30fd66ca8de52e4bf5671ecaa083eb7b92 "$repo/scripts/build_constgold_blend_lookup.py"
check_sha 52caed1c0d955693d0b0825688f4ef344d10f731b5b6813d146e153564b02db0 "$flow"
check_sha 9723589880234fa6825ae242d9b978120a6385faf5ccc9942d1993696952e46d "$emulator"
check_sha 48908a9cebc7475be83eef767c95175694531d78b1dd789c512fc49c0699f2b4 "$metadata"
check_sha dd01c7594b14f2dd598415cbb9c258438aacc0eaf91286d026d770f261d850d2 "$source_root/sbsi/models.py"
check_sha f1611699bb0200aea41d0ffff8675741883d0f5c01a7bd21f0f959b7639caaec "$source_root/sbsi/paths.py"
check_sha 76de8776a3702c0d159a5744aca8291c12ddef757b06fe69ddc1ac589bc4974d "$blendemu_root/blendemu/__init__.py"
check_sha 80dc9d028d6bf347e4397d4f9b58844cbd5bc944709b7fc5bdf6e268bc947149 "$blendemu_root/blendemu/catalog.py"
check_sha 0b9a2bb380c28cc7c6a0db32b0be65bc733023b61235ac6d2a25cdd224dc4f5a "$blendemu_root/blendemu/config.py"
check_sha 5968f46b6fdb916433651472056bc6bcd8ef4d47516160abf877cab42e2fd73b "$blendemu_root/blendemu/data_utils.py"
check_sha b18b2f358e48ece3319560c7fe88f616cb2ff8a545efb53e5cc808497e2e7743 "$blendemu_root/blendemu/inference.py"
check_sha 9f339d8c05d2d7ab70f4bd7796e951b0d1c72adeb8dff0f57f8c7cfd4dc2624b "$blendemu_root/blendemu/nz_utils.py"
check_sha 9a12e34dcfad1452254a6aa1091faafaab9025a2f51fe3323d06f7a754a8172c "$blendemu_root/blendemu/response.py"
check_sha fe68919df7c93dfc43697c26184e1007e9b7679a1e9bfeeab9ee582365983bd2 "$blendemu_root/blendemu/shape.py"
check_sha c66434468e93d2f95ba25ba0e4ba8fd6ce066fc8a598e4e7982144fa98532c16 "$blendemu_root/blendemu/utils.py"

[[ -s "$decision" && ! -L "$decision" ]] || { echo "missing decision: $decision" >&2; exit 2; }
if ! "$analysis_python" - "$decision" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(
    path.read_text(),
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant {value}")
    ),
)
if payload.get("passes_reused_structural_screen") is not True:
    raise SystemExit(1)
PY
then
  echo "MINIMAL_RBLEND_G005_LOOKUP_SKIPPED structural_gate=false"
  exit 0
fi

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
  "$decision" "$emulator" "$metadata" "$repo/scripts/build_constgold_blend_lookup.py" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

import blendemu
import numpy as np
import pyarrow.feather as pf
import sbsi

first, last = map(int, sys.argv[1:3])
catalogue_root, output, receipt, decision, emulator, metadata, builder = map(
    Path, sys.argv[3:]
)

def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()

def input_path(case: int, sign: str) -> Path:
    return (
        catalogue_root
        / f"case{case}_{sign}/real0/catalogues/input/gals_info_tile180.0_-0.5.feather"
    )

columns = [
    "index_input", "RA_input", "DEC_input", "Re_input",
    "sersic_n_input", "r_input",
]
result = pf.read_table(output).to_pandas()
expected_columns = {"case", "input_index", "R_blend"}
if set(result) != expected_columns or result.duplicated(["case", "input_index"]).any():
    raise RuntimeError("invalid lookup schema or duplicate keys")
if set(result["case"].unique()) != set(range(first, last + 1)):
    raise RuntimeError("lookup case set mismatch")
if not np.isfinite(result["R_blend"]).all():
    raise RuntimeError("lookup contains non-finite R_blend")

input_hashes = {}
rows_by_case = {}
for case in range(first, last + 1):
    plus_path = input_path(case, "0.05")
    minus_path = input_path(case, "-0.05")
    plus = pf.read_table(plus_path, columns=columns).to_pandas()
    minus = pf.read_table(minus_path, columns=columns).to_pandas()
    if len(plus) != len(minus):
        raise RuntimeError(f"truth-leg row-count mismatch: case={case}")
    for name in columns:
        if not np.array_equal(
            plus[name].to_numpy(), minus[name].to_numpy(), equal_nan=True
        ):
            raise RuntimeError(f"trial9 input differs between truth legs: case={case} column={name}")
    eligible = (
        np.isfinite(plus[["r_input", "Re_input"]]).all(axis=1)
        & (plus["r_input"] > 18.0)
        & (plus["r_input"] < 25.8)
        & (plus["Re_input"] > 0.37)
        & (plus["Re_input"] < 1.5)
    )
    expected_ids = np.sort(plus.loc[eligible, "index_input"].to_numpy(np.int64))
    actual_ids = np.sort(
        result.loc[result["case"] == case, "input_index"].to_numpy(np.int64)
    )
    if not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError(f"lookup support mismatch: case={case}")
    rows_by_case[str(case)] = len(actual_ids)
    input_hashes[str(plus_path)] = digest(plus_path)
    input_hashes[str(minus_path)] = digest(minus_path)

meta = json.loads(metadata.read_text())
if meta["tasks"]["regression"]["cuts"][1] != [18, 25.8] or meta["tasks"]["regression"]["cuts"][3] != [0.37, 1.5]:
    raise RuntimeError("trial9 support cuts changed")
if receipt.exists():
    raise RuntimeError(f"refusing to overwrite {receipt}")
payload = {
    "schema_version": 1,
    "analysis": "frozen trial9 lookup for existing g1=+/-0.05 catalogues",
    "cases": list(range(first, last + 1)),
    "truth_legs": [-0.05, 0.05],
    "trial9_inputs_equal_between_legs": True,
    "rows": len(result),
    "rows_by_case": rows_by_case,
    "environment": {
        "python": sys.executable,
        "sbsi": str(Path(sbsi.__file__).resolve()),
        "blendemu": str(Path(blendemu.__file__).resolve()),
    },
    "artifacts": {
        str(output): digest(output),
        str(decision): digest(decision),
        str(emulator): digest(emulator),
        str(metadata): digest(metadata),
        str(builder): digest(builder),
    },
    "input_hashes": input_hashes,
    "checks_passed": True,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
receipt.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
print(
    f"MINIMAL_RBLEND_G005_LOOKUP_READY cases={first}-{last} "
    f"rows={len(result)} sha256={digest(output)}",
    flush=True,
)
PY

sha256sum "$output" "$receipt"

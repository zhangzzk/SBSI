#!/usr/bin/env bash
# Reused cases120--139 response screen; no simulation or measurement.

#SBATCH --job-name=sbsi-ms-response
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_reused_response_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_reused_response_v1/logs/%x_%A_%a.err

set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_shape_reused_response_v1
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
lookup_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/selection_replication_c90_139_v1/prepared
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

if [[ "${PHASE:?Set PHASE=evaluate,review}" == review ]]; then
  cd "$repo"
  "$analysis_python" -u -m scripts.summarize_minimal_selected_shape_response review --root "$root"
  exit
fi

index=${SLURM_ARRAY_TASK_ID:?evaluation requires array index 0 or 1}
case "$index" in
  0) arm=baseline ;;
  1) arm=selected_shape ;;
  *) echo "invalid arm index: $index" >&2; exit 2 ;;
esac

flow=$("$analysis_python" - "$root" "$arm" <<'PY'
from scripts.summarize_minimal_selected_shape_response import load_protocol
import sys
print(load_protocol(sys.argv[1], verify=True)["arms"][sys.argv[2]])
PY
)
output=$root/response/$arm.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$analysis_python" - "$catalogue_root" "$lookup_root" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

catalogue_root, lookup_root = map(Path, sys.argv[1:])
def digest(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing, empty, or linked input: {path}")
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()

for case in range(120, 140):
    case_root = lookup_root / f"case{case}"
    receipt = json.loads((case_root / "receipt.json").read_text())
    if receipt.get("case") != case or receipt.get("checks_passed") is not True:
        raise RuntimeError(f"invalid existing receipt: case={case}")
    rblend = case_root / "rblend.feather"
    if digest(rblend) != receipt["hashes"][str(rblend)]:
        raise RuntimeError(f"R_blend changed: case={case}")
    paths = {
        catalogue_root / f"case{case}_{sign}" / "real0/catalogues" / relative
        for sign in ("-0.02", "0.02")
        for relative in (
            "input/gals_info_tile180.0_-0.5.feather",
            "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
            "Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
        )
    }
    if set(map(str, paths)) != set(receipt["catalogue_hashes"]):
        raise RuntimeError(f"catalogue receipt path set changed: case={case}")
    for path in paths:
        if digest(path) != receipt["catalogue_hashes"][str(path)]:
            raise RuntimeError(f"catalogue changed: {path}")
print("MINIMAL_SELECTED_SHAPE_EXISTING_INPUTS_VERIFIED", flush=True)
PY

cases=()
lookups=()
for case in $(seq 120 139); do
  cases+=(--case "$case")
  lookups+=(--rblend-lookup "$lookup_root/case${case}/rblend.feather")
done
detector_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_selected_event_head_v2/models

export PYTHONPATH="$repo:$source_root"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_selected_shape_response \
  --constgold-root "$catalogue_root" \
  "${lookups[@]}" \
  --crowd-lookup /project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather \
  --measurement-model "$flow" \
  --detection-classifier "$detector_root/seed20260913/selected.pt" \
  --detection-classifier "$detector_root/seed20260914/selected.pt" \
  --detection-classifier "$detector_root/seed20260915/selected.pt" \
  --probability-factorization direct_selected \
  --detection-neighbour-selection nearest --detection-impact-exponent 2 \
  --shear-component g1 "${cases[@]}" \
  --output "$output" --h 0.02 --draws 256 \
  --sampling-seed 7301 --sampling-seed 7302 \
  --sampling-seed 7303 --sampling-seed 7304 \
  --batch-size 2048 --device cuda \
  --n-boot 10000 --bootstrap-seed 20260914 \
  --disable-shape-cut

sha256sum "$output"
echo "MINIMAL_SELECTED_SHAPE_RESPONSE_DONE arm=$arm"

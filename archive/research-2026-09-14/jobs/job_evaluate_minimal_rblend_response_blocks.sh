#!/usr/bin/env bash
# Frozen four-endpoint response worker. Submit explicitly as 0-151 after all 38
# simulation-block receipts exist; task = 4 * block + endpoint.

#SBATCH --job-name=sbsi_mrblend_resp
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_fresh_acceptance_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_fresh_acceptance_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_fresh_acceptance_v1
production_root=$root/production
response_root=$root/response_blocks
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/paired/epoch154.pt
detector_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/models/flow8_rblend
detector1=$detector_root/seed20260913/selected.pt
detector2=$detector_root/seed20260914/selected.pt
detector3=$detector_root/seed20260915/selected.pt

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

: "${SLURM_ARRAY_TASK_ID:?submit with an explicit array index in 0..151}"
task=$SLURM_ARRAY_TASK_ID
(( task >= 0 && task < 152 )) || { echo "invalid task: $task" >&2; exit 2; }
block=$((task / 4))
endpoint=$((task % 4))
printf -v block_name 'block_%02d' "$block"

case "$endpoint" in
  0) group=g1_h002; component=g1; h=0.02; minus=-0.02; plus=0.02; minus_leg=g1_m02; plus_leg=g1_p02 ;;
  1) group=g1_h004; component=g1; h=0.04; minus=-0.04; plus=0.04; minus_leg=g1_m04; plus_leg=g1_p04 ;;
  2) group=g2_h002; component=g2; h=0.02; minus=-0.02; plus=0.02; minus_leg=g2_m02; plus_leg=g2_p02 ;;
  3) group=g2_h004; component=g2; h=0.04; minus=-0.04; plus=0.04; minus_leg=g2_m04; plus_leg=g2_p04 ;;
  *) echo "unreachable endpoint: $endpoint" >&2; exit 2 ;;
esac

check_sha e981c0d0516647aa4b5d0a02b60cde36ec61cece6909253e344af525393577c0 "$repo/configs/likelihood_minimal_rblend_candidate.json"
check_sha 1e5d4a81156cc33045cef2e6ef97112a1997aee67d52533cedbcb303587a1714 "$repo/scripts/evaluate_constgold_complete_forward_response.py"
check_sha be53879078164546056dbb8336b22226563cab17e5ca2bfa7e414d3007ba1686 "$repo/sbsi/catalogue_likelihood.py"
check_sha 0186af211f194471ab9fd1ebcf3ef5d9eaebd1a5ba4126b2440ebfbda2b91e37 "$repo/sbsi/coordinates.py"
check_sha 8534830d0c738ab91f14c791061f4d71cf32a913baed2a5c47cb432d7261c6ba "$repo/sbsi/flow_size_condition.py"
check_sha d154b06d20c3bb092eab998ca8e67d9149865c042f73d16b42c62dcf82bc894c "$repo/sbsi/measurement_model.py"
check_sha dd01c7594b14f2dd598415cbb9c258438aacc0eaf91286d026d770f261d850d2 "$repo/sbsi/models.py"
check_sha 7a440bf91c8bd6cbff52d3dd3b2646b6e489ce865561d4d4ef9229a594f45b7f "$repo/sbsi/scene_prior.py"
check_sha 4f77ad81cbb7ed4749fe10884a696f43262d38bc8ba545c548e9c269267980bb "$repo/sbsi/selection_model.py"
check_sha bc5b635de48768a297050820a71ac03fa26f5b18c3aea7c7fb05238156c0d9ca "$repo/sbsi/shear_map.py"
check_sha 9c5bb028437a4714b454d2bcd202243a2eda6263c36b4c70c46de3d12f725a6f "$flow"
check_sha bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685 "$detector1"
check_sha 031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd "$detector2"
check_sha dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702 "$detector3"

# Require the complete 760-case simulation stage, then rehash every retained
# catalogue used by this task against its selected block's receipt.
"$analysis_python" - "$production_root" "$block" "$group" "$minus" "$plus" "$minus_leg" "$plus_leg" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

production, selected, group, minus, plus, minus_leg, plus_leg = sys.argv[1:]
production = Path(production)
selected = int(selected)

def digest(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing, empty, or linked response input: {path}")
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()

receipts = []
for block in range(38):
    first = 1001 + 20 * block
    path = production / f"block_{block:02d}" / "block_receipt.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"all simulation blocks must finish before response: {path}")
    receipt = json.loads(path.read_text())
    expected = {
        "status": "validated_and_pruned",
        "protocol": "minimal-rblend-fresh-acceptance-v1",
        "block_index": block,
        "case_first": first,
        "case_last": first + 19,
        "case_count": 20,
        "leg_count": 9,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"invalid simulation receipt {path}: {key}")
    receipts.append(receipt)

receipt = receipts[selected]
block_root = production / f"block_{selected:02d}"
if digest(block_root / "rblend.feather") != receipt["rblend"]["sha256"]:
    raise RuntimeError("R_blend lookup changed after block validation")

paths = {
    "truth": ("input", "gals_info_tile180.0_-0.5.feather"),
    "noise": ("input", "noise_info_tile180.0_-0.5.csv"),
    "shape": ("Shapes", "shape_catalogue_detect_position_all_tile180.0_-0.5.feather"),
    "crossmatch": ("CrossMatch", "tile180.0_-0.5_rot0_matched.feather"),
}
for case in range(receipt["case_first"], receipt["case_last"] + 1):
    for token, leg in ((minus, minus_leg), (plus, plus_leg)):
        catalogue = block_root / group / f"case{case}_{token}" / "real0" / "catalogues"
        recorded = receipt["outputs"][str(case)][leg]["sha256"]
        for key, relative in paths.items():
            if digest(catalogue.joinpath(*relative)) != recorded[key]:
                raise RuntimeError(f"retained {key} changed after validation: case={case} leg={leg}")
PY

output_dir=$response_root/$group
output=$output_dir/$block_name.json
mkdir -p "$output_dir"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

cases=()
first=$((1001 + 20 * block))
for case_id in $(seq "$first" "$((first + 19))"); do
  cases+=(--case "$case_id")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root "$production_root/$block_name/$group" \
  --rblend-lookup "$production_root/$block_name/rblend.feather" \
  --measurement-model "$flow" \
  --detection-classifier "$detector1" \
  --detection-classifier "$detector2" \
  --detection-classifier "$detector3" \
  --detection-neighbour-selection nearest --detection-impact-exponent 2 \
  --shear-component "$component" \
  "${cases[@]}" \
  --output "$output" \
  --h "$h" --draws 64 \
  --sampling-seed 7301 --sampling-seed 7302 \
  --sampling-seed 7303 --sampling-seed 7304 \
  --batch-size 2048 --device cuda \
  --n-boot 2 --bootstrap-seed 20260913 \
  --disable-shape-cut

sha256sum "$output"
echo "MINIMAL_RBLEND_RESPONSE_BLOCK_DONE block=$block endpoint=$group output=$output"

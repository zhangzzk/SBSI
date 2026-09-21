#!/usr/bin/env bash
# Re-evaluate the frozen nine-input flow on existing cases120--139 only.

#SBATCH --job-name=sbsi_mrflow_oldresp
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/response_screen/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/response_screen/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
experiment=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
response_root=$experiment/response_screen
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
lookup_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/selection_replication_c90_139_v1/prepared
crowd_lookup=/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=$experiment/training/flow8_rblend/epoch016.pt
detector_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/models/flow8_rblend
detector1=$detector_root/seed20260913/selected.pt
detector2=$detector_root/seed20260914/selected.pt
detector3=$detector_root/seed20260915/selected.pt
output=$response_root/result.json

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

check_sha d1e7b1194f0ea75bc35e8d4c01479e60b225a204216cd384107dfc223dff73f0 "$repo/scripts/evaluate_constgold_complete_forward_response.py"
check_sha a680c82bdbc2e819a08c76f78ce5b72f2bf93f11db7badf15798258375887390 "$source_root/sbsi/catalogue_likelihood.py"
check_sha 0186af211f194471ab9fd1ebcf3ef5d9eaebd1a5ba4126b2440ebfbda2b91e37 "$source_root/sbsi/coordinates.py"
check_sha 3294e9b629413a00596927c2239c80ff978e09e886e352df5298444de5bfbd1b "$source_root/sbsi/flow_disk.py"
check_sha daaae1aa846bf898dfd8e5d55c4861eb83d87990b93f9dfed6f2e274c3ee4b08 "$source_root/sbsi/flow_physical_disk.py"
check_sha f82bee07da4aac6644ba66af5f27776c001eec2848bb3f7228b67d300bc726ad "$source_root/sbsi/flow_physical_response.py"
check_sha 8534830d0c738ab91f14c791061f4d71cf32a913baed2a5c47cb432d7261c6ba "$source_root/sbsi/flow_size_condition.py"
check_sha d154b06d20c3bb092eab998ca8e67d9149865c042f73d16b42c62dcf82bc894c "$source_root/sbsi/measurement_model.py"
check_sha dd01c7594b14f2dd598415cbb9c258438aacc0eaf91286d026d770f261d850d2 "$source_root/sbsi/models.py"
check_sha 7a440bf91c8bd6cbff52d3dd3b2646b6e489ce865561d4d4ef9229a594f45b7f "$source_root/sbsi/scene_prior.py"
check_sha 226cf6de61745dac557156820535e0e0d25ced06f156e44bc04b8ed0141205c5 "$source_root/sbsi/selection_model.py"
check_sha bc5b635de48768a297050820a71ac03fa26f5b18c3aea7c7fb05238156c0d9ca "$source_root/sbsi/shear_map.py"
check_sha 52caed1c0d955693d0b0825688f4ef344d10f731b5b6813d146e153564b02db0 "$flow"
check_sha bbbd27a86c18bffb3e8b98e46c455ebcd2a27844ead363913bb5289ae19e6685 "$detector1"
check_sha 031498aa4140be4d2dfc7605f5211a1d2379e14c16f3fbc23b79f2f7f9e028cd "$detector2"
check_sha dab1e4cddf3b61da4cc6dee810091209b29df4dbf5dcc86cd20d3e23eebbd702 "$detector3"
check_sha c7fd37f1144a4437e98b6008086aaec4d1f9ec77d3f674afa477d461eeecd712 "$crowd_lookup"

mkdir -p "$response_root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

# Revalidate the exact retained catalogue and R_blend bytes against the
# existing per-case receipts. This reads inputs but creates no catalogue.
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
        for chunk in iter(lambda: stream.read(8_388_608), b""):
            value.update(chunk)
    return value.hexdigest()

for case in range(120, 140):
    case_root = lookup_root / f"case{case}"
    receipt_path = case_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("case") != case or receipt.get("checks_passed") is not True:
        raise RuntimeError(f"invalid existing preparation receipt: {receipt_path}")
    rblend = case_root / "rblend.feather"
    if digest(rblend) != receipt["hashes"][str(rblend)]:
        raise RuntimeError(f"R_blend changed after preparation: case={case}")
    expected_paths = {
        catalogue_root / f"case{case}_{sign}" / "real0/catalogues" / relative
        for sign in ("-0.02", "0.02")
        for relative in (
            "input/gals_info_tile180.0_-0.5.feather",
            "CrossMatch/tile180.0_-0.5_rot0_matched.feather",
            "Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather",
        )
    }
    recorded = receipt["catalogue_hashes"]
    if set(map(str, expected_paths)) != set(recorded):
        raise RuntimeError(f"catalogue receipt path set mismatch: case={case}")
    for path in expected_paths:
        if digest(path) != recorded[str(path)]:
            raise RuntimeError(f"catalogue changed after preparation: {path}")
    print(f"MINIMAL_RBLEND_EXISTING_INPUT_VERIFIED case={case}", flush=True)
PY

cases=()
lookups=()
for case in $(seq 120 139); do
  cases+=(--case "$case")
  lookups+=(--rblend-lookup "$lookup_root/case${case}/rblend.feather")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$source_root:$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u "$repo/scripts/evaluate_constgold_complete_forward_response.py" \
  --constgold-root "$catalogue_root" \
  "${lookups[@]}" \
  --crowd-lookup "$crowd_lookup" \
  --measurement-model "$flow" \
  --detection-classifier "$detector1" \
  --detection-classifier "$detector2" \
  --detection-classifier "$detector3" \
  --detection-neighbour-selection nearest --detection-impact-exponent 2 \
  --shear-component g1 \
  "${cases[@]}" \
  --output "$output" \
  --h 0.02 --draws 64 \
  --sampling-seed 7301 --sampling-seed 7302 \
  --sampling-seed 7303 --sampling-seed 7304 \
  --sampling-seed 7305 --sampling-seed 7306 \
  --sampling-seed 7307 --sampling-seed 7308 \
  --sampling-seed 7309 --sampling-seed 7310 \
  --sampling-seed 7311 --sampling-seed 7312 \
  --sampling-seed 7313 --sampling-seed 7314 \
  --sampling-seed 7315 --sampling-seed 7316 \
  --batch-size 2048 --device cuda \
  --n-boot 10000 --bootstrap-seed 20260913 \
  --disable-shape-cut

sha256sum "$output"
echo "MINIMAL_RBLEND_FLOW_EXISTING_RESPONSE_DONE output=$output"

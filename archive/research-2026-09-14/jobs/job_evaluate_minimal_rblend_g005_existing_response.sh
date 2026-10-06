#!/usr/bin/env bash
# Evaluate the unchanged minimal candidate on existing cases140--239 only.

#SBATCH --job-name=sbsi_mrflow_g005
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1/existing_g005_c140_239/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
experiment=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_rblend_flow_condition_v1
root=$experiment/existing_g005_c140_239
catalogue_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239
decision=$experiment/response_screen/decision.json
crowd_lookup=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_g005_c140-239/crowd_flux_conc_c140-239.feather
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_grid_free_shape_radius_lambda10_v2/constgold50_v1/source
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
analysis_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=$experiment/training/flow8_rblend/epoch016.pt
detector_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/minimal_coherent_u_flow_rblend_ablation_v2/models/flow8_rblend
detector1=$detector_root/seed20260913/selected.pt
detector2=$detector_root/seed20260914/selected.pt
detector3=$detector_root/seed20260915/selected.pt
output=$root/response/result.json

check_sha() {
  local expected=$1
  local path=$2
  echo "$expected  $path" | sha256sum --check --status
}

[[ -s "$decision" && ! -L "$decision" ]] || { echo "missing decision: $decision" >&2; exit 2; }
if ! "$analysis_python" - "$decision" <<'PY'
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
  echo "MINIMAL_RBLEND_G005_RESPONSE_SKIPPED structural_gate=false"
  exit 0
fi

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
check_sha 959e1dc323f6d4f5aec05bcbfdf0ab10ee9027a57a445c701be478e823370824 "$crowd_lookup"

[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

# Rehash the complete lookup receipts before any response is evaluated.
"$analysis_python" - "$root" "$decision" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root, decision = map(Path, sys.argv[1:])

def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing, empty, or linked artifact: {path}")
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8_388_608), b""):
            value.update(block)
    return value.hexdigest()

decision_hash = digest(decision)
covered = []
for first in range(140, 240, 10):
    last = first + 9
    receipt_path = root / "lookup" / f"rblend_trial9_c{first}-{last}.json"
    receipt = json.loads(
        receipt_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value}")
        ),
    )
    cases = list(range(first, last + 1))
    if (
        receipt.get("checks_passed") is not True
        or receipt.get("cases") != cases
        or receipt.get("trial9_inputs_equal_between_legs") is not True
    ):
        raise RuntimeError(f"invalid lookup receipt: {receipt_path}")
    artifacts = receipt.get("artifacts", {})
    if artifacts.get(str(decision)) != decision_hash:
        raise RuntimeError(f"lookup used a different gate decision: {receipt_path}")
    for rendered, expected in {**artifacts, **receipt.get("input_hashes", {})}.items():
        path = Path(rendered)
        if digest(path) != expected:
            raise RuntimeError(f"artifact changed after lookup preparation: {path}")
    lookup = root / "lookup" / f"rblend_trial9_c{first}-{last}.feather"
    if str(lookup) not in artifacts:
        raise RuntimeError(f"receipt omits lookup hash: {receipt_path}")
    covered.extend(cases)
if covered != list(range(140, 240)):
    raise RuntimeError("lookup receipts do not cover cases140--239 exactly")
print("MINIMAL_RBLEND_G005_LOOKUPS_VERIFIED cases=140-239", flush=True)
PY

cases=()
for case in $(seq 140 239); do
  cases+=(--case "$case")
done
lookups=()
for first in $(seq 140 10 230); do
  last=$((first + 9))
  lookups+=(--rblend-lookup "$root/lookup/rblend_trial9_c${first}-${last}.feather")
done

mkdir -p "$root/response"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$source_root:$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /tmp
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
  --h 0.05 --draws 64 \
  --sampling-seed 7301 --sampling-seed 7302 \
  --sampling-seed 7303 --sampling-seed 7304 \
  --sampling-seed 7305 --sampling-seed 7306 \
  --sampling-seed 7307 --sampling-seed 7308 \
  --sampling-seed 7309 --sampling-seed 7310 \
  --sampling-seed 7311 --sampling-seed 7312 \
  --sampling-seed 7313 --sampling-seed 7314 \
  --sampling-seed 7315 --sampling-seed 7316 \
  --batch-size 2048 --device cuda \
  --n-boot 10000 --bootstrap-seed 20260914 \
  --disable-shape-cut

sha256sum "$output"
echo "MINIMAL_RBLEND_G005_EXISTING_RESPONSE_DONE output=$output"

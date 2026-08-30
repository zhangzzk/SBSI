#!/usr/bin/env bash
# Frozen centre-only exact-oracle gate for no-pilot H256-U8-PS2 integer-CDF DM3.
# Reuses the separately audited outcome-blind H256 partition; never builds or
# tunes a partition and never runs a pilot or adaptive allocation.

#SBATCH --job-name=sbsi_h256_u8_ps2_center
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
expected_commit=${SBSI_COMMIT:-}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_h256_u8_ps2_center_dev_v1}
partition_cache=${SPATIAL_H256_U8_PS2_PARTITION_CACHE:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1/partition_audit_v1}
compact="$source_root/compact_global"

OBSERVATION_START=25000
OBSERVATION_STOP=25512
K=16384
PREFILTER=131072
GLOBAL_DRAWS=16384
PROPOSAL_SEED=8701
TAG=rows25000_25512_seed8701_spatialH256_u8_ps2_dm3_m16384_exact64_v1
output="$root/results/$TAG"

if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
  echo "missing isolated repository checkout at $repo" >&2
  exit 2
fi
if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SBSI_COMMIT must pin one full lowercase 40-character commit" >&2
  exit 2
fi
actual_commit=$(git -C "$repo" rev-parse --verify HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "checkout commit $actual_commit does not match SBSI_COMMIT $expected_commit" >&2
  exit 2
fi
if git -C "$repo" symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "refusing a branch checkout; submit from a detached commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing a dirty repository checkout at $repo" >&2
  exit 2
fi
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" ]]; then
  echo "missing frozen mock at $source_mock" >&2
  exit 2
fi
if [[ ! -f "$partition_cache/spatial_partition.npz" || ! -f "$partition_cache/manifest.json" ]]; then
  echo "missing reviewed spatial partition under $partition_cache" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/results" "$root/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
export SBSI_SPATIAL_H256_U8_PS2_PARTITION_CACHE="$partition_cache"
export SBSI_SPATIAL_H256_U8_PS2_MOCK="$source_mock"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

from sbsi.catalogue_sampling import (
    SPATIAL_PS_CELLS_PER_SLAB,
    SPATIAL_PS_DRAWS_PER_CELL,
    SPATIAL_PS_HASH_CONTRACT,
    SPATIAL_PS_MICROSTRATA,
    SPATIAL_PS_ORDER_CONTRACT,
    SPATIAL_PS_QUANTIZER_CONTRACT,
    SPATIAL_PS_REPLICATES,
    SPATIAL_PS_RNG_CONTRACT,
    SPATIAL_PS_SEED,
    SPATIAL_PS_SOURCE_COUNTS,
    SPATIAL_PS_TICKS_PER_MICROSTRATUM,
    SPATIAL_PS_VERSION,
    SPATIAL_PARTITION_RECIPE_SHA256,
    TAIL_PROPOSAL_RECIPE,
)
from sbsi.complement_diagnostic import exact_oracle_object_ids
from sbsi.spatial_h256_u8_ps2_diagnostic import (
    SPATIAL_H256_U8_PS2_PROTOCOL,
)

protocol = SPATIAL_H256_U8_PS2_PROTOCOL
partition_root = Path(os.environ["SBSI_SPATIAL_H256_U8_PS2_PARTITION_CACHE"])
mock_root = Path(os.environ["SBSI_SPATIAL_H256_U8_PS2_MOCK"])

for name in (
    "partition_sha256",
    "partition_cache_sha256",
    "partition_manifest_sha256",
):
    value = protocol.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"H256-U8-PS2 protocol has no reviewed {name}")
if protocol["partition_recipe_sha256"] != SPATIAL_PARTITION_RECIPE_SHA256:
    raise SystemExit("spatial partition recipe changed")
if protocol["tail_proposal_recipe"] != asdict(TAIL_PROPOSAL_RECIPE):
    raise SystemExit("conditioned-DM3 recipe changed")
expected_core = {
    "sampler_version": SPATIAL_PS_VERSION,
    "sampler_hash_contract": SPATIAL_PS_HASH_CONTRACT,
    "rng_contract": SPATIAL_PS_RNG_CONTRACT,
    "quantizer_contract": SPATIAL_PS_QUANTIZER_CONTRACT,
    "order_contract": SPATIAL_PS_ORDER_CONTRACT,
    "production_seed": SPATIAL_PS_SEED,
    "cells_per_slab": SPATIAL_PS_CELLS_PER_SLAB,
    "cdf_ticks_per_microstratum": SPATIAL_PS_TICKS_PER_MICROSTRATUM,
    "source_microstrata": list(SPATIAL_PS_MICROSTRATA),
    "source_draws_per_cell": list(SPATIAL_PS_SOURCE_COUNTS),
    "draws_per_microstratum": SPATIAL_PS_REPLICATES,
    "draws_per_cell": SPATIAL_PS_DRAWS_PER_CELL,
}
observed_core = {name: protocol[name] for name in expected_core}
if observed_core != expected_core:
    raise SystemExit("H256-U8-PS2 core contract changed")
if protocol["pilot"] != {
    "used": False,
    "draws": 0,
    "target_flow_evaluations": 0,
    "rng_domains": [],
    "allocation": "fixed_uniform_8",
}:
    raise SystemExit("H256-U8-PS2 no-pilot identity changed")
_, selection_manifest = exact_oracle_object_ids(
    protocol["universe_start"], protocol["universe_stop"]
)
if selection_manifest != protocol["exact_selection_manifest_sha256"]:
    raise SystemExit("H256-U8-PS2 exact-oracle selection changed")

partition_path = partition_root / "spatial_partition.npz"
manifest_path = partition_root / "manifest.json"
cache_hash = sha256(partition_path.read_bytes()).hexdigest()
manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
if cache_hash != protocol["partition_cache_sha256"]:
    raise SystemExit("spatial partition cache bytes changed")
if manifest_hash != protocol["partition_manifest_sha256"]:
    raise SystemExit("spatial partition manifest bytes changed")
manifest = json.loads(manifest_path.read_text())
if manifest.get("partition_sha256") != protocol["partition_sha256"]:
    raise SystemExit("spatial partition semantic identity changed")
if manifest.get("partition_cache_sha256") != cache_hash:
    raise SystemExit("spatial partition manifest does not identify its cache")
if manifest.get("partition_recipe_sha256") != protocol["partition_recipe_sha256"]:
    raise SystemExit("spatial partition manifest recipe changed")
if manifest.get("cell_count_minimum", 0) <= 16_384:
    raise SystemExit("spatial partition has an undersized cell")

for name, expected_hash in protocol["mock_input_sha256"].items():
    actual_hash = sha256((mock_root / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"H256-U8-PS2 frozen mock changed for {name}: "
            f"{actual_hash} != {expected_hash}"
        )
PREFLIGHT

exec "$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --mock-input "$source_mock" \
  --output "$output" \
  --observation-start "$OBSERVATION_START" \
  --observation-stop "$OBSERVATION_STOP" \
  --proposal-candidates "$K" \
  --proposal-prefilter-candidates "$PREFILTER" \
  --proposal-seed "$PROPOSAL_SEED" \
  --initial-strategy mean_observed_shape \
  --candidate-backend torch \
  --object-chunk 32 \
  --atom-chunk 4096 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  --diagnose-complement-exact-oracle \
  --diagnose-complement-spatial-h256-u8-ps2 \
  --spatial-h256-u8-ps2-partition-cache "$partition_cache" \
  --spatial-h256-u8-ps2-object-chunk 32 \
  --spatial-h256-u8-ps2-atom-chunk 262144 \
  --device cuda

#!/usr/bin/env bash
# Frozen centre-only exact-oracle gate for spatial discarded-pilot Neyman DM3.
# The outcome-blind partition must first be audited, reviewed, and frozen by
# all three protocol hashes; this launcher never builds or tunes it.

#SBATCH --job-name=sbsi_spatial_neyman_center
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
# Slurm opens these paths before the job can create an absolute log directory.
# Keep them relative to the directory from which this script is submitted.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
expected_commit=${SBSI_COMMIT:-}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1}
partition_cache=${SPATIAL_NEYMAN_PARTITION_CACHE:-$root/partition_audit_v1}
compact="$source_root/compact_global"

OBSERVATION_START=25000
OBSERVATION_STOP=25512
K=16384
PREFILTER=131072
GLOBAL_DRAWS=16384
PROPOSAL_SEED=8701
TAG=rows25000_25512_seed8701_spatialH256_pilot4096_dm3_m16384_exact64_v1
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
export SBSI_SPATIAL_PARTITION_CACHE="$partition_cache"
export SBSI_SPATIAL_MOCK="$source_mock"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

from sbsi.catalogue_sampling import (
    SPATIAL_PARTITION_RECIPE_SHA256,
    TAIL_PROPOSAL_RECIPE,
)
from sbsi.complement_diagnostic import exact_oracle_object_ids
from sbsi.spatial_neyman_diagnostic import SPATIAL_NEYMAN_PROTOCOL

protocol = SPATIAL_NEYMAN_PROTOCOL
partition_root = Path(os.environ["SBSI_SPATIAL_PARTITION_CACHE"])
mock_root = Path(os.environ["SBSI_SPATIAL_MOCK"])

for name in (
    "partition_sha256",
    "partition_cache_sha256",
    "partition_manifest_sha256",
):
    value = protocol.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"spatial protocol has no reviewed {name}")
if protocol["partition_recipe_sha256"] != SPATIAL_PARTITION_RECIPE_SHA256:
    raise SystemExit("spatial partition recipe changed")
if protocol["tail_proposal_recipe"] != asdict(TAIL_PROPOSAL_RECIPE):
    raise SystemExit("spatial conditioned-DM3 recipe changed")
_, selection_manifest = exact_oracle_object_ids(
    protocol["universe_start"], protocol["universe_stop"]
)
if selection_manifest != protocol["exact_selection_manifest_sha256"]:
    raise SystemExit("spatial exact-oracle selection changed")

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
if manifest.get("cell_count_maximum", 262_145) > 262_144:
    raise SystemExit("spatial partition has an oversized cell")

for name, expected_hash in protocol["mock_input_sha256"].items():
    actual_hash = sha256((mock_root / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"spatial frozen mock changed for {name}: "
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
  --object-chunk 8 \
  --atom-chunk 4096 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  --diagnose-complement-exact-oracle \
  --diagnose-complement-spatial-neyman \
  --spatial-neyman-partition-cache "$partition_cache" \
  --spatial-neyman-object-chunk 8 \
  --spatial-neyman-atom-chunk 262144 \
  --device cuda

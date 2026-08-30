#!/usr/bin/env bash
# Frozen exact-oracle confirmation for the conditioned DM4 tail proposal.
#
# The 512-object universe, 64-object exact subset, hash key, atom/object chunks,
# and every promotion gate are code constants, not launcher knobs.  This wrapper
# reserves fresh rows 25,512:26,024 and a production-depth M=131,072 screen.

#SBATCH --job-name=sbsi_tail_dm_exact_oracle
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_tail_dm_mis_heldout25k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_tail_dm_mis_heldout25k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_tail_dm_mis_heldout25k_v1}
compact="$source_root/compact_global"

OBSERVATION_START=25512
OBSERVATION_STOP=26024
K=16384
PREFILTER=131072
GLOBAL_DRAWS=131072
PROPOSAL_SEED=9701
TAG=rows25512_26024_seed9701_dm4_exact64_m131072_v1
output="$root/results/$TAG"

for variable in K PREFILTER GLOBAL_DRAWS PROPOSAL_SEED; do
  value=${!variable}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$variable must be an unsigned integer, got $value" >&2
    exit 2
  fi
done
if (( K == 0 || PREFILTER < K )); then
  echo "invalid candidate sizes" >&2
  exit 2
fi
if (( GLOBAL_DRAWS == 0 || GLOBAL_DRAWS % 8 != 0 )); then
  echo "GLOBAL_DRAWS must be a positive multiple of the fixed 8-draw cycle" >&2
  exit 2
fi

if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
  echo "missing isolated repository checkout at $repo" >&2
  exit 2
fi
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" ]]; then
  echo "missing frozen mock at $source_mock" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/results" "$root/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
from hashlib import sha256
from pathlib import Path

from sbsi.complement_diagnostic import (
    EXACT_ORACLE_CONFIRMATION,
    exact_oracle_object_ids,
)

_, manifest = exact_oracle_object_ids(25512, 26024)
expected = "fc44a632ed4758440a3e241ffeab9fba087f58397f114cfbc38aecc7feedcb8b"
if manifest != expected:
    raise SystemExit(f"exact-oracle selection changed: {manifest} != {expected}")
mock = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/"
    "inference_candidate_ladder_n125k_v1/prepare_cont303/mock"
)
for name, expected_hash in EXACT_ORACLE_CONFIRMATION["mock_input_sha256"].items():
    actual_hash = sha256((mock / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"exact-oracle source changed for {name}: {actual_hash} != {expected_hash}"
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
  --candidate-backend torch \
  --object-chunk 64 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  --diagnose-complement-tail-proposal \
  --diagnose-complement-exact-oracle \
  --device cuda

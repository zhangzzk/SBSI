#!/usr/bin/env bash
# Development-only complement diagnostic for the fixed independent-pilot
# exact-stratum recipe.  The discovery rows are already consumed; this wrapper
# must never be pointed at the frozen exact-oracle confirmation window.

#SBATCH --job-name=sbsi_tail_pilot_stratum
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_pilot_stratum_dev_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_pilot_stratum_dev_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_pilot_stratum_dev_v1}
compact="$source_root/compact_global"

OBSERVATION_START=25000
OBSERVATION_STOP=25512
K=16384
PREFILTER=131072
GLOBAL_DRAWS=16384
PROPOSAL_SEED=8701
TAG=rows25000_25512_seed8701_pilot8192_promote4096_meanobs_dm3_v1
output="$root/results/$TAG"

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
from sbsi.catalogue_sampling import (
    PILOT_STRATUM_RECIPE,
    TAIL_PROPOSAL_ATOM_CHUNK,
    TAIL_PROPOSAL_OBJECT_CHUNK,
    TAIL_PROPOSAL_RECIPE,
)
from sbsi.complement_diagnostic import (
    COMPLEMENT_DIAGNOSTIC_SEED_OFFSET,
    PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET,
)

if PILOT_STRATUM_RECIPE.pilot_draws != 8192:
    raise SystemExit("pilot draw count changed")
if PILOT_STRATUM_RECIPE.n_promotions != 4096:
    raise SystemExit("pilot promotion count changed")
if PILOT_STRATUM_RECIPE.pilot_proposal != TAIL_PROPOSAL_RECIPE:
    raise SystemExit("pilot and production DM3 recipes differ")
if PILOT_STRATUM_RECIPE.atom_chunk != TAIL_PROPOSAL_ATOM_CHUNK:
    raise SystemExit("pilot atom chunk changed")
if PILOT_STRATUM_RECIPE.object_chunk != TAIL_PROPOSAL_OBJECT_CHUNK:
    raise SystemExit("pilot object chunk changed")
if PILOT_STRATUM_RECIPE.sha256 != (
    "b4349271763311c2127d785c79fc637e115f421a31fb821e6db6e0a592b8d410"
):
    raise SystemExit("pilot recipe hash changed")
if PILOT_STRATUM_DIAGNOSTIC_SEED_OFFSET != 1_000_003:
    raise SystemExit("pilot seed offset changed")
if COMPLEMENT_DIAGNOSTIC_SEED_OFFSET != 7_000_003:
    raise SystemExit("production diagnostic seed offset changed")
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
  --object-chunk 64 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  --diagnose-complement-tail-proposal \
  --diagnose-complement-pilot-stratum \
  --device cuda

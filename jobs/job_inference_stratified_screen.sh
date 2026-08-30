#!/usr/bin/env bash
# Fresh estimator screen for the fixed conditioned deterministic-MIS tail.
#
# Rows through 26,023 are reserved for proposal development and confirmation.
# The default 26,024:51,024 window is a disjoint 25,000-row estimator screen.
# Run paired arms by changing only MODE:
#
#   MODE=stratified sbatch jobs/job_inference_stratified_screen.sh
#   MODE=tilted_stratified sbatch jobs/job_inference_stratified_screen.sh
#
# `tilted_stratified` has no tunable tail-proposal flags. Its component
# temperatures, allocation, and prior defense are the code-owned fixed recipe
# recorded in result.json. No wrapper value may silently change them.
# Repeat both arms with the same nondefault PROPOSAL_SEED for the independent-
# seed gate; the seed is included in the default output tag.

#SBATCH --job-name=sbsi_tail_dm_heldout
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_tail_dm_mis_heldout25k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_tail_dm_mis_heldout25k_v1/logs/%x_%j.err

set -euo pipefail

# This explicit default is intentional: the screen must run the isolated
# checkout, never whichever branch happens to occupy the original repository.
repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${SCREEN_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_tail_dm_mis_heldout25k_v1}
compact="$source_root/compact_global"

: "${MODE:?set MODE to stratified or tilted_stratified}"
case "$MODE" in
  stratified|tilted_stratified) ;;
  *) echo "MODE must be stratified or tilted_stratified, got $MODE" >&2; exit 2 ;;
esac

OBSERVATION_START=${OBSERVATION_START:-26024}
OBSERVATION_STOP=${OBSERVATION_STOP:-51024}
K=${K:-16384}
PROPOSAL_SEED=${PROPOSAL_SEED:-8701}
TAG=${TAG:-${MODE}_rows${OBSERVATION_START}_${OBSERVATION_STOP}_seed${PROPOSAL_SEED}_dmdet_t1_t2_pdet_prior_3_2_2_1_v1}
output="$root/results/$TAG"

for variable in OBSERVATION_START OBSERVATION_STOP K PROPOSAL_SEED; do
  value=${!variable}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$variable must be an unsigned integer, got $value" >&2
    exit 2
  fi
done
if (( OBSERVATION_STOP <= OBSERVATION_START || K == 0 )); then
  echo "invalid observation window or candidate size" >&2
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
# py31 still lacks the required editable install (doc/ENVIRONMENT.md). Keep
# the explicit checkout first until that environment defect is repaired.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
from sbsi.catalogue_sampling import TAIL_PROPOSAL_RECIPE  # noqa: F401
PREFLIGHT

args=(
  --inference-config "$repo/configs/inference.json"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$compact/scene_store"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --model-cache "$compact/model_cache"
  --proposal-cache "$compact/proposal_cache"
  --mock-input "$source_mock"
  --output "$output"
  --observation-start "$OBSERVATION_START"
  --observation-stop "$OBSERVATION_STOP"
  --proposal-candidates "$K"
  --proposal-seed "$PROPOSAL_SEED"
  --estimator-mode "$MODE"
  --device cuda
)

exec "$python" "$repo/scripts/run_inference.py" "${args[@]}"

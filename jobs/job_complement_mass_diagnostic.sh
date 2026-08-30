#!/usr/bin/env bash
# Complement diagnostic for the fixed DM-MIS tail recipe.
#
# The default 25,000:25,512 window is the development screen used to choose the
# broad-bridge family. A confirmation run must override it with a fresh
# predeclared window and unique tag. The diagnostic compares the prior
# complement estimator with the exact production sampler and records each
# component's pre-conditioning complement mass. The recipe has no command-line
# temperatures or weights.
# Jobs 16121540--16121542 failed before startup because the retired wrapper
# reused Slurm's reserved TEMP scratch-path variable as a float. This wrapper
# deliberately exposes no temperature variable; any future diagnostic-only
# override must be named TILT_TEMPERATURE and validated as numeric.

#SBATCH --job-name=sbsi_tail_dm_diagnostic
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
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_tail_dm_mis_heldout25k_v1}
compact="$source_root/compact_global"

OBSERVATION_START=${OBSERVATION_START:-25000}
OBSERVATION_STOP=${OBSERVATION_STOP:-25512}
K=${K:-16384}
PREFILTER=${PREFILTER:-131072}
GLOBAL_DRAWS=${GLOBAL_DRAWS:-16384}
PROPOSAL_SEED=${PROPOSAL_SEED:-8701}
TAG=${TAG:-rows${OBSERVATION_START}_${OBSERVATION_STOP}_seed${PROPOSAL_SEED}_dmdet_t1_t2_pdet_prior_3_2_2_1_v1}
output="$root/results/$TAG"

for variable in OBSERVATION_START OBSERVATION_STOP K PREFILTER GLOBAL_DRAWS PROPOSAL_SEED; do
  value=${!variable}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$variable must be an unsigned integer, got $value" >&2
    exit 2
  fi
done
if (( OBSERVATION_STOP <= OBSERVATION_START || K == 0 || PREFILTER < K )); then
  echo "invalid observation window or candidate sizes" >&2
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
import sbsi, blendemu  # noqa: F401
from sbsi.catalogue_sampling import TAIL_PROPOSAL_RECIPE  # noqa: F401
from sbsi.complement_diagnostic import diagnose_complement  # noqa: F401
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
  --device cuda

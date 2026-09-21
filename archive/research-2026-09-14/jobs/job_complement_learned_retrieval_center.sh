#!/usr/bin/env bash
# Frozen development-only exact learned-R plus conditioned-DM3 centre gate.
# A failure stops the proposed nine-view S/B/O bridge before implementation.

#SBATCH --job-name=sbsi_learned_R_center
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
# Keep Slurm's initial log paths relative to the existing submission directory.
# The job cannot create an absolute log parent before Slurm opens these files.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
checkpoint=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_contrastive_proposal_b2048_s20000_v1/pilot/model.pt
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_learned_retrieval_center_dev_v1}
compact="$source_root/compact_global"

OBSERVATION_START=25000
OBSERVATION_STOP=25512
K=524288
GLOBAL_DRAWS=16384
PROPOSAL_SEED=8701
TAG=rows25000_25512_seed8701_learnedR524288_dm3_m16384_exact64_v1
output="$root/results/$TAG"

if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
  echo "missing isolated repository checkout at $repo" >&2
  exit 2
fi
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" ]]; then
  echo "missing frozen mock at $source_mock" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "missing frozen learned checkpoint at $checkpoint" >&2
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
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from sbsi.catalogue_sampling import TAIL_PROPOSAL_RECIPE
from sbsi.complement_diagnostic import exact_oracle_object_ids
from sbsi.learned_retrieval_diagnostic import (
    LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256,
    LEARNED_RETRIEVAL_CENTER_PROTOCOL,
)

protocol = LEARNED_RETRIEVAL_CENTER_PROTOCOL
if asdict(TAIL_PROPOSAL_RECIPE) != protocol["tail_proposal_recipe"]:
    raise SystemExit("learned-centre DM3 recipe changed")
_, selection_manifest = exact_oracle_object_ids(
    protocol["universe_start"], protocol["universe_stop"]
)
if selection_manifest != protocol["exact_selection_manifest_sha256"]:
    raise SystemExit("learned-centre exact selection changed")
checkpoint = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/"
    "infer_v1_contrastive_proposal_b2048_s20000_v1/pilot/model.pt"
)
actual_checkpoint = sha256(checkpoint.read_bytes()).hexdigest()
if actual_checkpoint != LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256:
    raise SystemExit(
        f"learned checkpoint changed: {actual_checkpoint} != "
        f"{LEARNED_RETRIEVAL_CENTER_CHECKPOINT_SHA256}"
    )
mock = Path(
    "/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/"
    "inference_candidate_ladder_n125k_v1/prepare_cont303/mock"
)
for name, expected_hash in protocol["mock_input_sha256"].items():
    actual_hash = sha256((mock / name).read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise SystemExit(
            f"learned-centre source changed for {name}: "
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
  --proposal-prefilter-candidates "$K" \
  --proposal-seed "$PROPOSAL_SEED" \
  --initial-strategy mean_observed_shape \
  --candidate-backend torch \
  --object-chunk 8 \
  --atom-chunk 4096 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  --diagnose-complement-tail-proposal \
  --diagnose-complement-exact-oracle \
  --diagnose-complement-learned-center-checkpoint "$checkpoint" \
  --learned-retrieval-object-chunk 8 \
  --learned-retrieval-atom-chunk 262144 \
  --device cuda

#!/usr/bin/env bash
# Does the estimate converge in the proposal's candidate support K?
#
# cont.304 established that the draw ladder does NOT converge at 500k: ghat1
# climbs monotonically through every rung to M=16,384, the 4,096->16,384 paired
# pull is +4.47 sigma, and the top rung has no plateau above it.  cont.305 then
# showed the drift is a small coherent push on essentially every object -- mean
# per-object score1 rises 42% while its scatter and kurtosis stay flat, and the
# 100 largest movers contribute about 0% of the total change.  So cont.304's
# open question ("do rare objects drive the continuing high-M shift?") is
# answered: they do not.  A uniform bias of that shape is a property of the
# estimator, not of a tail.
#
# Throughout that ladder the candidate pool was held FIXED at K=16,384 while M
# ran to 16,384, and at the top rung the unique-atom count is 1,567/5,779/10,602
# (min/median/max).  The sampler is resampling a bounded support ever more
# finely.  Two facts point at K rather than M as the binding constraint:
#
#   1. unique atoms saturate well below K as M grows;
#   2. the cont.297 slice measured cutting K from 16,384 to 4,096 moving ghat1
#      by +8.9e-3, a far larger lever than any draw-budget change.
#
# This job sweeps K at fixed M and retains the full nested M ladder, giving a
# 5 (K) x 6 (M) grid on ONE frozen mock under common random numbers.  If ghat1
# drifts with K as it drifts with M, the proposal does not span the space and
# more draws only sample a too-narrow support more finely.  If ghat1 flattens in
# K, the draw-budget drift is a separate defect and K is exonerated.
#
# Submit with K and TAG exported:
#   for k in 8192 16384 32768 65536 131072; do
#     K=$k TAG=k$k sbatch --job-name=sbsi_cand_k$k jobs/job_inference_candidate_ladder.sh
#   done
#
# RELEASE LABELLING.  Every arm overrides `proposal.candidates` on the command
# line, so `_resolved_pipeline_config` will differ from `configs/inference.json`
# and each manifest is labelled `pipeline_release: "custom"` rather than
# `v1.1-infer`.  That is correct and intended: this is a deliberate sweep around
# the release point, not the release point itself.  Nothing else is overridden,
# so the K=16,384 arm differs from the base config in no field at all.
#
# VALIDATION ANCHOR.  Because the K=16,384 arm is the base configuration, it is
# checkable rather than merely plausible.  Reducing the cont.304 per-object
# moments to rows [0,125000) at the same global centre predicts what it must
# return:
#
#       M        ghat1        ghat2         se1        se2
#     512    +0.018025    -0.000629    4.74e-04   3.78e-04
#    1024    +0.018887    -0.000638    5.27e-04   3.97e-04
#    2048    +0.019689    -0.000674    5.89e-04   4.18e-04
#    4096    +0.020653    -0.000815    6.66e-04   4.48e-04
#    8192    +0.020945    -0.000754    6.56e-04   4.58e-04
#   16384    +0.021898    -0.000919    8.73e-04   4.87e-04
#
# Agreement to float32 reassociation confirms the harness under the cont.303
# restructure.  Disagreement means this wrapper differs from cont.304 in a way
# not intended here, and every other arm must be discarded until that is
# explained.  Note this is also the first inference result produced through
# `scripts/run_inference.py` on this mock, so the anchor doubles as a
# cont.303 migration check against the retired `job_infer_v1.sh` path.
#
# N=125,000 keeps the sweep affordable and is ample: the drift under test spans
# 3.9e-3 across the M ladder against a robust se near 5-9e-4.
#
# The initial centre is the raw mean measured shape over the WHOLE mock
# (n_objects=500000 in the cont.300/304 identity blocks), not over the observed
# window, so restricting to [0,125000) does not move the expansion point.

#SBATCH --job-name=sbsi_inference_candidate_ladder
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
# The cont.299 frozen 500k mock.  Reused, never regenerated: a fresh mock would
# answer a different, multi-catalogue question and would discard the common
# random numbers that make this sweep paired (doc/CONVENTIONS.md section 6d).
# The cont.299 mock regenerated under cont.303 code by job 16090436
# (jobs/job_prepare_stacked_n500k_mock.sh).  The original copy is still rejected
# by `_validate_loaded_likelihood_manifest`: its provenance schema predates the
# restructure, so the identity dict cannot match however unchanged the data are.
# This regeneration reproduces the original measurements.parquet and
# truth.parquet SHA-256 byte-for-byte, so it is the same mock with current
# paperwork -- the common random numbers, and therefore pairing against the
# cont.304/305 ladder and the anchor table below, are preserved.
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1
compact="$source_root/compact_global"

: "${K:?set the candidate pool size}"
: "${TAG:?set the arm output tag}"

if (( K > 131072 )); then
  echo "K=$K exceeds the 131072 prefilter; raise the prefilter deliberately" >&2
  exit 2
fi

output="$root/results/$TAG"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" ]]; then
  echo "missing frozen mock at $source_mock" >&2
  exit 2
fi
mkdir -p "$root/results"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# doc/ENVIRONMENT.md#installation asks for an editable install and forbids
# putting checkouts on PYTHONPATH.  Neither `sbsi` nor `blendemu` is actually
# installed into the py31 environment, so that contract is currently unmet and
# `scripts/run_inference.py` cannot import its own package: the first
# submission of this sweep (16089180-16089184) died in 4-10 s on
# `ModuleNotFoundError: No module named 'sbsi'`.  The retired `job_infer_v1.sh`
# masked this by exporting PYTHONPATH; the cont.303 replacement
# `jobs/job_inference.sh` does not, and fails the same way.  Export it here so
# the sweep can run, and fix the environment separately -- this line should be
# deleted once py31 carries a real editable install.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

# Fail in seconds on a broken environment rather than after a GPU allocation.
"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
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
  --observation-start 0 --observation-stop 125000 \
  --proposal-candidates "$K" \
  --device cuda

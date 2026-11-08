#!/bin/bash
# Exact whole-catalogue integrand for a handful of observations (cont.331).
#
# Every convergence number reported so far -- Pareto tail index, ESS, maximum
# weight fraction -- is a property of `c_j / q_j` and therefore of the sampler
# as much as of the target.  The tail index correlates with measured magnitude
# at Spearman rho = -0.726: 39% of the brightest decile fail the k < 0.7 test
# against about 1% of the faint half.  That is consistent with two opposite
# stories, and the diagnostics in hand cannot separate them:
#
#   * the target is fine and the proposal simply misses the atoms that matter
#     for bright galaxies, which is a sampling problem and worth more sampler
#     work; or
#   * a bright galaxy has small measurement errors, so its likelihood is sharp
#     and only a handful of the 12.76 million atoms contribute at all.  Then no
#     proposal can help and the finite scene prior is too coarse for those
#     objects, which is a catalogue problem wearing a sampler's clothes.
#
# This job removes the proposal from the question by scoring every active atom
# exactly for each selected observation.  ROWS are positions in the frozen
# 25,000-object window, so they line up with the per-object diagnostics in
# `inference_stratified_screen_v1/results/tilted_batched/one_step_moments.npz`.
#
# Cost is one flow evaluation per atom per observation: 12,760,990 each, so a
# dozen observations is about 2% of one 25,000-object inference run.  It does
# not scale past a handful and is not meant to.
#
# The defaults reproduce the original self-response landscape.  For a mock
# whose manifest declares the complete likelihood, also set INFERENCE_CONFIG,
# BLEND_CACHE, CUT_ABS_EHAT and CUT_BOUND.  Every supplied term is checked
# against that manifest before whole-catalogue scoring begins.
#
#SBATCH --job-name=sbsi_landscape
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/likelihood_landscape_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/likelihood_landscape_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
inference_config=${INFERENCE_CONFIG:-$repo/configs/inference.json}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
# The same frozen mock the cont.317/329/330 screen uses, so a landscape can be
# matched to the tail index that motivated selecting that object.
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${LANDSCAPE_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/likelihood_landscape_v1}
compact="$source_root/compact_global"

: "${TAG:?set the output tag}"
: "${ROWS:?set ROWS to a space-separated list of window positions}"
OBSERVATION_STOP=${OBSERVATION_STOP:-25000}
HEAD=${HEAD:-4096}

output="$root/results/$TAG"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" ]]; then
  echo "missing frozen mock at $source_mock" >&2
  exit 2
fi
mkdir -p "$root/results" "$root/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# cont.307: `import sbsi` succeeds only from the repository root in py31.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

extra_args=()
if [[ -n "${BLEND_CACHE:-}" ]]; then
  extra_args+=(--blend-response-cache "$BLEND_CACHE")
fi
if [[ -n "${CUT_ABS_EHAT:-}" ]]; then
  extra_args+=(--cut-abs-ehat "$CUT_ABS_EHAT")
fi
if [[ -n "${CUT_BOUND:-}" ]]; then
  extra_args+=(--cut-bound "$CUT_BOUND")
fi

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
from sbsi.likelihood_landscape import likelihood_landscape  # noqa: F401
PREFLIGHT

exec "$python" "$repo/scripts/run_inference.py" \
  --inference-config "$inference_config" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --mock-input "$source_mock" \
  --output "$output" \
  --observation-start 0 --observation-stop "$OBSERVATION_STOP" \
  --candidate-backend torch \
  --device cuda \
  --likelihood-landscape "$output" \
  --likelihood-landscape-head "$HEAD" \
  --proposal-candidates "${K:-16384}" \
  --proposal-prefilter-candidates "${PREFILTER:-131072}" \
  "${extra_args[@]}" \
  ${SHORTLISTS:+--likelihood-landscape-shortlists $SHORTLISTS} \
  --likelihood-landscape-rows $ROWS

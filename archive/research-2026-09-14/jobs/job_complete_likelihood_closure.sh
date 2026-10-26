#!/usr/bin/env bash
# Closure against the COMPLETE catalogue likelihood: flow + detection
# classifier + external R_blend + a measured selection cut.
#
# WHY NOT run_inference.py.  The production estimator refuses both terms
# outright -- sbsi/catalogue_null.py raises "adaptive Section 5 currently
# requires measured cuts disabled" and "adaptive Section 5 currently requires
# external R_blend=0", and the importance-autograd path raises the same pair.
# Those guards are correct: the tensor-native fast paths reuse a zero-shear
# tensor base and shortcut the shear dependence, which a per-atom blend shift
# and a shear-dependent B_W(g) both invalidate
# (`tensor_shape_only_available` already returns False once R_blend is
# attached).  The likelihood itself supports both terms -- the mock is drawn
# from it -- so the closure runs through the validation entry point instead.
#
# scripts/run_catalogue_closure.py --sampler exact scores every observed object
# against all 12,760,990 prior atoms with no sampler in the loop, so it answers
# "does the complete likelihood close?" without entangling that question with
# the separate sampling-algorithm search.  N_DETECTED is small by necessity:
# the exact sum is O(N_obj x N_atom) per shear evaluation.
#
# NO --detection-model.  That flag routes to load_detection_classifier, which
# torch.loads a checkpoint; the v3.2-like detection model is the blendemu
# classification JSON, which the emulator loads itself.  Omitting the flag
# takes the load_emulator branch -- the same detector run_inference.py uses.
#
# NO --model-cache.  compact_global/model_cache was written by
# merge_sharded_prior_model_qmc.py with its own metadata schema, and the
# closure script compares a cache's stored identity for full equality against
# one it would have written itself, so the two can never match.  Reusing it
# would also buy little: it is stored as "shape_only_zero_base" with a single
# g=0 view, and CatalogueLikelihood.tensor_shape_only_available is False once
# R_blend is attached, so every non-zero stencil shear has to be recomputed
# either way.  The views are built in memory instead.
#
# The script re-derives the mock's stored r_blend from the supplied cache at
# rtol=0, atol=0 and refuses any mock row outside the declared cut, so a mock
# and a likelihood that disagree cannot be scored against each other.
#
#   N_DETECTED=200 sbatch jobs/job_complete_likelihood_closure.sh

#SBATCH --job-name=sbsi_complete_closure_exact
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
blend=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/blend_response_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_complete_likelihood_v1

TAG=${TAG:-exact_n${N_DETECTED:-200}}
output="$root/closure/$TAG"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/logs" "$root/closure"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

exec "$python" "$repo/scripts/run_catalogue_closure.py" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --emulator-model "$repo/models/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json" \
  --emulator-metadata "$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json" \
  --inject-r-blend \
  --blend-response-cache "$blend" \
  --selection-cache "$root/closure_selection_cache" \
  --cut-abs-ehat 0.6 \
  --cut-bound "measured_mag_auto::25.8" \
  --output "$output" \
  --sampler exact \
  --n-detected "${N_DETECTED:-200}" \
  --injected-g1 0.02 \
  --injected-g2 0.0 \
  --score-center-g1 0.0 \
  --score-center-g2 0.0 \
  --direction-g1 1.0 \
  --direction-g2 0.0 \
  --detection-radius-arcsec 3.0 \
  --flow-neighbour-radius-arcsec 7.0 \
  --crowding-near-arcsec 3.0 \
  --crowding-far-arcsec 7.0 \
  --pixel-size 0.2 \
  --zero-point 30.0 \
  --psf-fwhm 0.73 \
  --moffat-beta 2.224 \
  --pixel-rms 0.312 \
  --device cuda

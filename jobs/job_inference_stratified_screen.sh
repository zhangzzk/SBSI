#!/usr/bin/env bash
# Does splitting the sum, instead of blending the proposal, remove the M drift?
#
# cont.309 established the shape of the defect: ghat1 converges in the candidate
# pool K but not in the draw budget M, the two drifts are independent, and the
# M increments do not halve at any pool size.  It closed by moving suspicion to
# the estimator and asking for weight diagnostics.
#
# Two earlier measurements say what the estimator is actually doing, and they
# have not been read together before:
#
#   cont.264 (exact 12.76M-atom scan, observation 514716): the K=16,384
#     candidates hold 62.4% of the target mass but receive 90.0% of the
#     proposal mass.  The remaining 37.6% is carried entirely by the epsilon=0.1
#     defensive draws.  Asymptotic ESS/M is 6.69e-4 -- about 11 effective draws
#     at M=16,384 -- with chi-square divergence 1,493 and maximum p/q 138,252.
#     Capture plateaus in K (62.4% at 16,384, 65.4% at 1,048,576), so enlarging
#     the pool cannot fix it.
#
#   cont.283 (64 rows): Infer V1 target capture is p10/median/p90
#     5.004/25.118/82.561%.  The stress row above is BETTER than typical.  For
#     the median object roughly three quarters of the likelihood is estimated by
#     the ~1,638 global draws while 90% of the compute goes to the quarter that
#     could have been summed exactly.
#
# Read together these explain cont.309 rather than conflicting with it.  ghat
# converging in K is not reassurance: it confirms the ranker has saturated, at
# which point everything rests on the epsilon tail -- which is precisely the
# part that does not converge in M.  It also predicts that
# `bias_correction=richardson_1_over_m` cannot help, because a 1/M log-bias
# assumes finite weight variance and chi-square ~1,500 with a 1e5 maximum ratio
# is the opposite of that.
#
# THE CHANGE UNDER TEST.  `--estimator-mode stratified` stops blending and
# splits the sum instead:
#
#     A_i = sum_{j in S_i} c_j  +  (1/M) sum_t 1[j_t not in S_i] c_jt / pi_jt
#
# The candidate stratum is summed exactly and contributes no variance at all;
# every draw then estimates only the complement.  The algebra is not a new
# approximation -- the epsilon component of the mixture is already equivalent to
# a complement estimator with M_tail = epsilon*M draws, the epsilon cancelling
# in pi/(epsilon*pi).  So this arm changes the allocation from 0.1*M tail draws
# to M tail draws, a ~10x variance reduction on the term that carries most of
# the mass, and removes the dominant stratum's variance entirely.
#
# `draw_stratified` reads its atom from the same uniform column that
# `draw_adapted` uses for its global component, so at one seed the retained
# complement draws ARE the mixture's own global draws: the two arms are paired
# by common random numbers (doc/CONVENTIONS.md section 6d), on one frozen mock.
#
# COST.  Per object per stencil view the mixture evaluates ~unique(M) atoms
# (7,747 at M=16,384, K=65,536 in cont.309); this arm evaluates K exact plus
# ~M_tail unique, so at K=16,384 it is roughly 4x more flow calls.  That is why
# the default window is smaller than cont.309's 125,000.  Both arms use the same
# window, so the comparison is unaffected.
#
# WHAT DECIDES IT.  Both arms retain the full nested M ladder, so `result.json`
# carries ghat per rung directly.  If the diagnosis is right the stratified arm's
# ladder should be far flatter in M than the mixture arm's, and its K dependence
# should largely vanish.  If its ladder drifts just as much, the tail proposal
# itself -- not the allocation -- is the defect, and the next step is a capped
# without-replacement tail estimator rather than more draws.
#
# Both arms also now carry per-rung weight diagnostics -- ESS, maximum weight
# share, relative standard error, and the generalized-Pareto tail index k --
# per object in `one_step_moments.npz` and as percentiles under
# `weight_diagnostics` in `result.json`.  cont.309 asked for these; without
# them the heavy-tail reading was unmeasurable from any saved artifact.  Two
# cautions when reading them:
#
#   * Compare the arms on relative standard error, NOT on ESS/M.  The
#     stratified arm's exact stratum carries no variance and consumes no draws,
#     so its ESS fraction is mechanically lower even where it is far more
#     accurate.  On the unit toy the stratified ESS/M is 0.30 against the
#     mixture's 0.92 while its relative error is 3.5x SMALLER.
#   * k >= 0.5 means infinite weight variance, so no root-M rate exists and
#     more draws cannot rescue that arm.  cont.264 measured chi-square ~1,493
#     and a maximum ratio of 138,252 on one row, which predicts k near or above
#     1 for the mixture.  If the stratified arm's tail k stays there too, the
#     allocation was never the binding constraint.
#
# RELEASE LABELLING.  The stratified arm records `estimator_mode` in its
# resolved config, so `_resolved_pipeline_config` deviates from
# its baseline configuration and that arm is labelled `pipeline_release:
# "custom"`.  That is correct and intended: this is a screen, not a release.
# The mixture arm overrides nothing and must come back labelled `v1.1-infer`,
# which is why this screen names `configs/inference_v1_1.json` rather than
# the default configuration: that default is now v1.3-infer and declares a
# stratified mode of its own.
#
# Submit both arms:
#   for m in mixture stratified; do
#     MODE=$m TAG=$m sbatch --job-name=sbsi_strat_$m \
#       jobs/job_inference_stratified_screen.sh
#   done
#
# Read them with, from the repository root:
#   python scripts/compare_estimator_arms.py \
#     --arm mixture=$ROOT/results/mixture \
#     --arm stratified=$ROOT/results/stratified
# where ROOT is the `root` set below.  That checks the pairing before it reports
# anything, differences the arms with the common draws kept, and prints the
# weight diagnostics on the comparable column.

#SBATCH --job-name=sbsi_inference_stratified_screen
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_stratified_screen_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_stratified_screen_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
# The cont.299 frozen 500k mock as regenerated under cont.303 code by job
# 16090436.  Reused, never regenerated: a fresh mock would discard the common
# random numbers that make this screen paired, and would also break pairing
# against the cont.304/305/309 ladders.
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${SCREEN_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_stratified_screen_v1}
compact="$source_root/compact_global"

: "${MODE:?set MODE to mixture or stratified}"
: "${TAG:?set the arm output tag}"

# cont.344: `release` runs a named release document with no overrides at all,
# so the recorded `pipeline_release` is the release's own name rather than
# `custom`.  Every other MODE keeps overriding the v1.1 baseline as before.
case "$MODE" in
  mixture|stratified|tilted_stratified|release) ;;
  *) echo "MODE must be mixture, stratified, tilted_stratified or release, got $MODE" >&2; exit 2 ;;
esac

INFERENCE_CONFIG=${INFERENCE_CONFIG:-$repo/configs/inference_v1_1.json}
if [[ "$MODE" == release ]]; then
  for forbidden in K LADDER DRAWS TILT_DELTA; do
    if [[ -n "${!forbidden:-}" ]]; then
      echo "MODE=release takes its settings from $INFERENCE_CONFIG; unset $forbidden" >&2
      exit 2
    fi
  done
fi

# Both arms share the pool and the window; only the estimator differs.
K=${K:-16384}
OBSERVATION_STOP=${OBSERVATION_STOP:-25000}

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

# doc/ENVIRONMENT.md#installation asks for an editable install and forbids
# putting checkouts on PYTHONPATH.  That contract is still unmet in py31 --
# `import sbsi` succeeds only from the repository root, which is the cont.307
# defect -- so export it here as cont.306 did, and delete this line once py31
# carries a real editable install.
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

# Fail in seconds on a broken environment rather than after a GPU allocation.
"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

args=(
  --inference-config "$INFERENCE_CONFIG"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$compact/scene_store"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --model-cache "$compact/model_cache"
  --proposal-cache "$compact/proposal_cache"
  --mock-input "$source_mock"
  --output "$output"
  --device cuda
)
if [[ -z "${OBJECT_SUBSET:-}" ]]; then
  args+=(--observation-start 0 --observation-stop "$OBSERVATION_STOP")
fi
if [[ "$MODE" != release ]]; then
  args+=(--proposal-candidates "$K")
fi

# The mixture arm must override nothing, so that it reproduces the release
# point exactly and anchors the comparison.
if [[ "$MODE" != mixture && "$MODE" != release ]]; then
  args+=(--estimator-mode "$MODE")
fi
# cont.317: the tilted arm draws its complement from the whole-catalogue
# Gaussian-proxy proposal instead of the flat prior.  Same window, same seed,
# same uniform stream as the `stratified` arm, so the two remain paired.
if [[ "$MODE" == tilted_stratified ]]; then
  args+=(--estimator-tilt-delta "${TILT_DELTA:-0.1}")
fi
# cont.324: every object currently receives the deepest rung whatever its own
# relative error, which ranges from 0.011 to 0.817.  `independent_pilot` spends
# a separate 512-draw stream first and then gives each object the shallowest
# rung meeting the ESS and maximum-weight thresholds.  The pilot stream is
# independent of the production one, so the chosen rung is fixed before any
# production weight is inspected and no optional-stopping bias is introduced.
# cont.326: the ladder tops out at 16,384 for every object, and cont.325 found
# that proportional allocation wants MORE than that for the hard tail, which is
# where 77.8% of the summed variance sits.  LADDER extends it upward so the
# question "do deeper budgets actually help those objects" can be measured
# rather than assumed.  The rungs stay nested, so the retained ladder still
# reports every shallower rung from the same run.
if [[ -n "${LADDER:-}" ]]; then
  args+=(--adaptive-draw-ladder $LADDER)
fi
# cont.330: the retained ladder shows the shear error improving only 3.5% from
# the 512-draw rung to the 16,384-draw rung, because the population error is
# set by shape noise and not by the Monte Carlo.  Cost, in contrast, scales
# with the deepest rung.  DRAWS makes the production budget settable so the
# cost/precision trade can be measured instead of assumed; LADDER must be set
# with it so the retained rungs stay nested inside the budget.
if [[ -n "${DRAWS:-}" ]]; then
  args+=(--draws "$DRAWS")
fi

# cont.334: run the same scene against one half of the catalogue's active
# support.  Halves 0 and 1 partition the same permutation, so the paired
# difference measures how much of the answer comes from which atoms happen to
# be in the catalogue -- a term no reported error bar currently carries.
if [[ -n "${CATALOGUE_HALF:-}" ]]; then
  args+=(--catalogue-half "$CATALOGUE_HALF" --catalogue-half-seed "${CATALOGUE_HALF_SEED:-0}")
fi

# cont.337: precision of the two whole-catalogue inner products behind the
# tilted proposal, which cont.328 measured at 78% of estimator wall clock.
if [[ -n "${TILT_PRECISION:-}" ]]; then
  args+=(--tilt-score-precision "$TILT_PRECISION")
fi

# cont.339: the estimator takes ONE Newton step from the raw mean observed
# shape.  cont.204 showed the marginalized likelihood is parabolic near its
# maximum but not over the interval from the centre to it, so a single step
# across that interval mis-sizes systematically.  Overriding the centre turns
# the estimator into a fixed-point test: started at the true shear, a correct
# likelihood must not move.
if [[ -n "${INITIAL_G1:-}" ]]; then
  args+=(--initial-strategy fixed --initial "${INITIAL_G1},${INITIAL_G2:-0.0}")
fi
# cont.345: K and M were both measured powerless on the 673 objects that carry
# 43% of the summed variance -- a 16x deeper shortlist moved their median
# relative error only 0.0966 -> 0.0855, and their error falls as M^-0.227
# instead of the bulk's M^-0.465.  OBJECT_SUBSET restricts a run to a named set
# of absolute mock rows so that population can be pushed to a much deeper M on
# its own, cheaply, and the extrapolation tested rather than assumed.
if [[ -n "${OBJECT_SUBSET:-}" ]]; then
  args+=(--object-subset "$OBJECT_SUBSET")
fi
# cont.347: every run through this job so far left `blend_response` null and
# declared no measured cut, so R_blend and blend_shift were identically zero
# and B_W(g) normalized an unselected population.  doc/CONVENTIONS.md section 1
# defines R_model = R_flow + R_blend, so those runs measured the SELF response
# only.  BLEND_CACHE and the CUT_* variables switch the full likelihood on.
# They travel together on purpose: the selection cache identity binds
# blend_response_sha256, so a cache built without R_blend cannot be reused with
# one, and run_inference.py refuses a mock whose likelihood identity differs
# from the one used to evaluate it.
if [[ -n "${BLEND_CACHE:-}" ]]; then
  args+=(--blend-response-cache "$BLEND_CACHE")
fi
if [[ -n "${CUT_ABS_EHAT:-}" ]]; then
  args+=(--cut-abs-ehat "$CUT_ABS_EHAT")
fi
if [[ -n "${CUT_BOUND:-}" ]]; then
  # Space-separated NAME:LO:HI specs, each passed as its own --cut-bound.
  for spec in $CUT_BOUND; do
    args+=(--cut-bound "$spec")
  done
fi
if [[ -n "${SELECTION_CACHE:-}" ]]; then
  args+=(--selection-cache "$SELECTION_CACHE")
fi

if [[ -n "${ALLOCATION:-}" ]]; then
  args+=(--adaptive-allocation "$ALLOCATION")
  args+=(--adaptive-pilot-draws "${PILOT_DRAWS:-512}")
  args+=(--adaptive-pilot-safety-factor "${PILOT_SAFETY:-1.0}")
fi

exec "$python" "$repo/scripts/run_inference.py" "${args[@]}"

#!/usr/bin/env bash
# Where does the leftover target mass -- and the heavy weight tail -- actually sit?
#
# cont.313 showed the stratified estimator is better but not working: it cuts
# relative standard error 2.46x and the median Pareto tail index from +2.01 to
# +0.74, yet 54% of objects still sit above the 0.7 reliability threshold.  Its
# complement is drawn from the detected prior pi, whose ratio c_j/pi_j is
# unbounded because pi knows nothing about the observation.  Replacing that
# draw is the next lever, and there are two candidate designs that differ only
# in where the mass is:
#
#   * A third stratum over P \ S -- inside the 131,072-atom location prefilter,
#     outside the 16,384 atoms evaluated exactly -- sampled with the diagonal
#     Gaussian proxy that the reranker already computes over P and discards.
#     Cheap, but only useful if the complement mass and tail live there.
#
#   * A whole-catalogue tail proposal, which is far more expensive and needs a
#     per-atom score for all 12.76M atoms.
#
# Two prior measurements pull in opposite directions and neither settles it.
# cont.264 found capture plateaus in K (62.4% at 16,384, 65.4% at 1,048,576),
# and cont.278 found a 64x deeper prefilter bought only +2.8pp -- both suggest
# the mass is far out.  But those measure top-K *capture*, not where the
# complement's variance comes from, and the stratified estimator's variance is
# driven by the largest ratios, which plausibly sit just outside the boundary.
#
# The pilot record of the deleted cont.284 retriever settles a related question
# and is worth stating because it reframes the whole search.  In
# infer_v1_contrastive_proposal_b2048_s20000_v1/pilot/result.json the median
# asymptotic ESS fraction is 0.21% for today's diagonal ranker, 0.31% for the
# learned retriever, and 0.86% for `oracle_full_topk` -- exact top-K selection
# by the true target.  The oracle's median capture is 37.7%.  So even perfect
# retrieval into K=16,384 leaves ~62% of the median object's mass outside the
# support.  Retrieval is not the binding constraint and no ranker can become
# one; the complement estimator is the whole problem.
#
# THE MEASUREMENT.  `--diagnose-complement` evaluates the exact unnormalised
# target c_j = pi_j Pdet_j L_j on every atom of the 131,072 prefilter, so
# sum_S c and sum_{P\S} c are exact, not sampled.  It then draws an independent
# prior sample per object to estimate the mass outside P, and reports the
# Pareto tail index of that prior-sampled complement separately on its P\S and
# outside-P parts.  The estimator never runs.
#
# WHAT DECIDES IT.  If the P\S share of the complement mass is large and its
# tail index is the heavy one while the outside-P part is mild, the cheap third
# stratum is the right next change.  If the outside-P part carries the mass and
# the tail, the prefilter-restricted design is dead before it is written and
# the next step is a whole-catalogue tail proposal.
#
# Submit:
#   sbatch jobs/job_complement_mass_diagnostic.sh
# Read $root/summary.json, or the per-object table at $root/per_object.parquet.

#SBATCH --job-name=sbsi_complement_mass
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_mass_diagnostic_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_mass_diagnostic_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
# The same frozen mock the cont.309/313 ladders use, so the objects diagnosed
# here are the objects whose tail indices those runs reported.
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_mass_diagnostic_v1}
compact="$source_root/compact_global"

TAG=${TAG:-default}
K=${K:-16384}
PREFILTER=${PREFILTER:-131072}
GLOBAL_DRAWS=${GLOBAL_DRAWS:-131072}
OBSERVATION_STOP=${OBSERVATION_STOP:-512}
# Set TILT to a defensive fraction to also draw the complement from the
# whole-catalogue Gaussian-proxy proposal and report its tail index.
TILT=${TILT:-}
# Above one, flatten the tilted component to broaden its coverage.
# Not named TEMP: the compute nodes already export that as a scratch path,
# which silently shadowed the override and failed job 16121540.
TILT_TEMPERATURE=${TILT_TEMPERATURE:-1.0}
# Set EXACT_TOP to also report the tilted complement once that many
# whole-catalogue-score-ranked atoms are summed exactly instead of sampled.
# cont.322 measured that the objects still failing the tail test are the ones
# where the tilted proposal is TOO concentrated -- median 3,274 unique atoms
# out of 16,384 draws, one draw carrying 8.9% of the weight -- which is the
# signature of a few dominant atoms the candidate shortlist never offered.
EXACT_TOP=${EXACT_TOP:-0}

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

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
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
  --observation-start 0 --observation-stop "$OBSERVATION_STOP" \
  --proposal-candidates "$K" \
  --proposal-prefilter-candidates "$PREFILTER" \
  --candidate-backend torch \
  --object-chunk 64 \
  --diagnose-complement "$output" \
  --diagnose-complement-global-draws "$GLOBAL_DRAWS" \
  ${TILT:+--diagnose-complement-tilt "$TILT"} \
  ${TILT:+--diagnose-complement-tilt-temperature "$TILT_TEMPERATURE"} \
  ${EXACT_TOP:+--diagnose-complement-exact-top "$EXACT_TOP"} \
  --device cuda

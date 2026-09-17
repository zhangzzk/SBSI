# Likelihood and inference

## Current identities

SBSI has one inference workflow with independent numerical and likelihood
identities:

| Role | Identity | Configuration |
| --- | --- | --- |
| Default estimator | `v1.1-infer` | `configs/inference.json` |
| Tilted-stratified estimator | `v1.2-infer` | `configs/inference_v1_2.json` |
| Default likelihood | `v3.2-like` | `configs/likelihood.json` |
| Development model set | `V3.5-like` | `configs/likelihood_v3_5_like.json` |

`V3.5-like` is not the default and has not demonstrated 0.3% calibration. It
names lambda10 epoch154, trial9 `R_blend`, and the three-seed nine-input
coherent-`U` probability ensemble.

## Likelihood

Let finite scene atoms be `z_j` with prior masses `pi_j`. For observation
`y_i`, shear `g`, usable probability `q_j(g)`, and joint measurement density
`f_j(y_i|g,U)`, the unnormalized selected likelihood is

```text
L_i(g) = sum_j pi_j q_j(g) f_j(y_i | g, U).
```

Measured selection changes the population normalization:

```text
P_sel(g) = sum_j pi_j q_j(g)
           Pr[(MAG_AUTO, FLUX_RADIUS, e1, e2) pass | g, U, z_j].
```

The selected likelihood is `L_i(g)/P_sel(g)`. For V3.5-like, the measured cut
uses `MAG_AUTO < 25.8`, convolved SExtractor `FLUX_RADIUS >= 0.75 arcsec`, and
no measured-`|e|` cut. Magnitude and radius stay in the joint four-output flow;
they are not replaced by a direct selected-event classifier. The V3.5-like
config pins the equivalent physical-flow bounds (3.75 pixels and
flux-from-magnitude above 47.8630092322638) and therefore enables population
normalization by default.

The external response contribution is applied per atom and per shear view:

```text
e_model,j(g) = e_flow,j(g)
             + R_blend,j [e_truth,j(g) - e_truth,j(0)].
```

The classifier, flow, `R_blend` cache, and measured selection must share the
same atom keys and conditioning convention.

## Numerical estimators

### v1.1-infer

The default uses a defensive mixture proposal, a 131,072-atom prefilter,
16,384 candidates, and nested draw ladder
`512,1024,2048,4096,8192,16384`. It computes the two-dimensional score and
observed information at the declared expansion point and applies one full 2D
Newton step.

### v1.2-infer

The alternative keeps 1,024 high-probability atoms as an exact stratum and
samples the complement from a tilted whole-catalogue proposal. Its production
draw ladder stops at 8,192. The exact stratum is summed once and the sampled
complement remains importance weighted; no extra finite-draw correction is
applied.

The active tilted-stratified path retains every observation and uses the fixed
draw budget. The word `trimmed` in its configuration description does not mean
objects are removed. `minimum_ess` and `maximum_weight_fraction` control stopping
in the mixture path; they do not provide an adaptive stopping guarantee for
this path. Both configurations request one Newton update, not an iterative
convergence loop.

Both estimators retain nested draws and common random numbers. Configuration,
not the release string, controls behavior.

The score, information, selected-population derivatives, and Bartlett checks
are derived in [`MATH.md`](MATH.md).

## Active workflow

### 1. Prepare the finite scene prior

The active preparation commands are:

```text
scripts/assemble_fs2_scene_catalogue.py
scripts/extract_scene_catalogue.py
scripts/build_scene_prior.py
scripts/finalize_fs2_sharded_prior.py
scripts/audit_fs2_prior_generation.py
```

The resulting store contains every positive-mass primary plus any zero-mass
neighbour needed by a configured model view. Guard radii and source-case
separation are validated before inference.

### 2. Build fixed model caches

```text
scripts/build_catalogue_blend_response.py
scripts/build_sharded_blend_response.py
scripts/build_selection_normalization_cache.py
scripts/combine_selection_normalization_shards.py
scripts/build_sharded_prior_model_qmc.py
scripts/merge_sharded_prior_model_qmc.py
```

Each cache records scene, model, target, feature, selection, shear-grid,
implementation, and random-stream identities. A mismatch is a hard error.

For a new model on the compact prior, use
`scripts/build_model_cache_from_zero_view.py` with the original compaction
report and neighbour-complete zero-view cache. It verifies source hashes and
row alignment, reuses only deterministic truth/crowding features, and evaluates
new classifier probabilities and flow-QMC proposal coordinates. Build a new
atom-aligned response with `scripts/build_sharded_blend_response.py` when the
response emulator changes. The compact scene has no neighbour graph and cannot
reconstruct these features by itself.

### ConstGold observation input

`scripts/prepare_constgold_inference_input.py` consumes existing raw ConstGold
plus-leg truth, crossmatches, and measured shape tables. It joins explicit
identities, counts unmatched/unusable rows, applies the configured prior's
truth-domain bounds and the likelihood configuration's measured cuts, then samples without
replacement. No minus-leg detection/selection or measured-ellipticity cut is
applied. The resulting image-input manifest pins source hashes, row identities,
the flow checkpoint, selection, and sample seed. This is input preparation
from existing measurements; it does not simulate images or measure shapes.

To expand the active prior while keeping the same complete source scenes,
`scripts/prepare_prior_domain_cache.py` validates full-shard truth features and
row identities, selects the declared primary domain, and recomputes R_blend
with the full neighbour graphs. Its compact output preserves only verified
deterministic zero-shear features. `build_model_cache_from_zero_view.py` then
evaluates fresh classifier probabilities and flow-QMC coordinates on that
support. The [0.37/0.5-arcsec ConstGold rerun](INFERENCE_CONSTGOLD_RE037_SIZE050.md#authorized-setup)
records a three-full-pass example with normalization rebuilt at each centre.

### 3. Run and combine inference

```bash
python scripts/run_inference.py \
  --inference-config configs/inference.json \
  --likelihood-config configs/likelihood_v3_5_like.json \
  --scene-store /path/to/scene \
  --measurement-model /path/to/epoch154.pt \
  --blend-response-cache /path/to/rblend-cache \
  --model-cache /path/to/model-cache \
  --proposal-cache /path/to/proposal-cache \
  --output /new/output
```

Large runs may be partitioned and then reduced with
`scripts/combine_inference_partitions.py`. Hybrid carry-forward runs use
`scripts/combine_hybrid_pass.py`. Final human-readable reduction uses
`scripts/summarize_inference.py`.

The job wrappers retained in `jobs/` contain scheduler resources and explicit
runtime paths only; scientific defaults remain in configuration files.

## Output contract

Every inference output records:

- numerical and likelihood identities;
- effective configuration and implementation hashes;
- scene, flow, classifier, response, normalization, and proposal identities;
- observation window and object IDs;
- finite-difference centre and stencil;
- nested draw ladder and random streams;
- score, information, diagnostics, and final estimate.

Partition combination sums sufficient statistics before solving. It refuses
overlaps, gaps, mixed centres, mixed model identities, or missing row identity.

The reported robust standard error uses per-object score/information residuals
for the one-step estimator. It treats the expansion point, trained models,
finite prior, selection quadrature, and proposal random streams as fixed. It
does not by itself establish total uncertainty or repeated-catalogue coverage.
See the [500k ConstGold process review](INFERENCE_CONSTGOLD_500K_REVIEW.md#what-the-original-uncertainty-computes)
for case resampling and the [numerical controls](INFERENCE_CONSTGOLD_500K_REVIEW.md#numerical-convergence-and-tail-diagnostics)
for limitations of that run.

## V3.5-like status and acceptance boundary

The V3.5-like cached mechanism screen on cases120--139 gave
`m=-0.372 +/- 0.223%`. Its truth-population and measurement-residual
contributions were `-0.549pp` and `+0.177pp`, so the central value depended on
opposing errors and was not accepted. The actual-`U` oracle across cases40--139
gave `m=-0.129 +/- 0.125%`; its approximate 95% interval extending slightly
beyond `-0.3%` is acceptable for routing work to the classifier, not for final
promotion.

A final claim requires a frozen candidate evaluated on fresh multi-axis image
simulations, with the fixed 0.75-arcsec/no-`|e|` sample, numerical-QMC control,
and componentwise anti-cancellation gates. No response offset, `R_blend`
rescaling, temperature fit, threshold tuning, or result pasted from a reused
endpoint is allowed.

Detailed historical derivations and experiment branches are preserved in
`archive/research-2026-09-14/doc/INFERENCE_FULL.md` and the archived work log.

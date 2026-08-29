# Catalogue-prior inference

This document defines the finite-scene likelihood, its reusable caches, and the
scalable importance sampler. Estimator derivations are in `doc/INFERENCE.md`
and `doc/MATH.md`; measured validation history stays in `doc/WORKLOG.md`.

## Current release boundary

SBSI has one production inference path:

- `v1.1-infer` is the numerical pipeline defined by
  `configs/inference.json`;
- `v3.2-like` is its catalogue likelihood, defined by
  `configs/likelihood.json`;
- `scripts/run_inference.py` is the runner; and
- `jobs/job_inference.sh` is the cluster-local reference wrapper.

The releases are independent. A pipeline release identifies sampling and
solving; a likelihood release identifies the density, detector, response
artifact, target order, geometry, and shear transform. Neither string is a code
dispatch key.

`v3.2-like` uses one externally supplied seed-501 original-E measurement flow,
the bundled seven-input spin-0 BlendEMU classifier, and the bundled response
artifact used to construct an optional atom-aligned `R_blend` cache. It is not
the path-only `get_model("V3.2")` preset. See `models/README.md` for the
artifact identities and frozen hashes.

## Finite-scene likelihood

Let `z_j` be a scene-prior atom with mass `pi_j`, `S_g` the configured
shape-only shear map, and `xhat_i` one retained measurement. With optional
measured-output cut `W`, SBSI evaluates

```text
A_i(g) = sum_j pi_j P_det(S_g z_j)
                  p_flow(xhat_i - b_j(g) | S_g z_j)

Ppass_j(g) = Integral W(xhat + b_j(g))
                      p_flow(xhat | S_g z_j) dxhat

B_W(g) = sum_j pi_j P_det(S_g z_j) Ppass_j(g)

p(xhat_i | detected, W, g) = A_i(g) / B_W(g)

b_j(g) = R_blend,j [e(S_g z_j) - e(z_j)]
```

Every retained observation already satisfies `W(xhat_i)=1`, so the cut does
not add a numerator factor. It does not cancel from the selected-population
normalization `B_W`. `CatalogueSelection` estimates `Ppass_j` with common
random numbers across shear views.

`R_blend,j` is evaluated once at zero shear from the atom's complete supported
neighbour scene and then held fixed. Mock generation adds `b_j(g)`, likelihood
evaluation subtracts it before calling the flow, and measured-selection draws
add it before applying `W`. All three paths use the same atom-aligned array.

The measurement flow is conditioned on successful detection and usable
four-output measurement. The current classifier predicts detection, not the
combined usable event. A precision selected-image analysis therefore needs a
canonical usable-event model; measurement failure must not be represented as a
continuous output cut because the flow has no failure mass.

## Scene store

`scripts/build_scene_prior.py` converts an explicit user truth catalogue into:

- `galaxies.parquet` — truth rows and prior masses;
- `neighbours.npz` — the directed guarded neighbour graph; and
- `manifest.json` — schema version, build report, input identity, geometry,
  and user-supplied provenance.

The guard radius must cover every downstream model aperture. The graph is fixed
under the project shear convention: intrinsic ellipticity changes, while flux,
size, position, separation, and neighbour membership do not. Magnification and
positional shear are outside this release.

Only positive-mass rows can be sampled as primaries. Zero-mass rows remain in
the scene so they can contribute neighbour context. Duplicate handling and
unmatched rows are explicit and reported.

The scene produces separate model views because the learned components have
different support:

- the flow uses one primary plus configured radial crowding summaries;
- the detector uses its declared representative neighbour and aperture; and
- the response emulator uses every supported pair passing its own recorded
  cuts and scaling.

For `v3.2-like`, detector metadata must contain only the declared seven spin-0
features before its zero-shear probabilities may be reused. Unknown or
shape-dependent feature sets take the conservative recomputation path.

## Derived caches

Caches are performance artifacts, not scientific inputs. Every cache is
validated against the complete identity needed to interpret it.

### Model views

`CatalogueModelCache` stores atom-aligned flow conditions, detection
probabilities, and optional blend shifts for requested shear views. The current
flow changes only its two primary intrinsic-shape columns; all other declared
conditions are reusable under the shape-only transform.

### Blend response

`scripts/build_catalogue_blend_response.py` evaluates the response emulator
once and writes the fixed atom-aligned `R_blend` cache. Its identity includes
the scene store, emulator model and metadata, observing conditions, and pair
configuration. An atom with no supported pair has a physical zero response;
missing or misaligned keys are errors.

### Proposal coordinates

`ProposalCoordinateTable` summarizes QMC draws of the complete flow for each
active atom in measured-output space. It does not rely on an optional explicit
mean head. The manifest records target names, location and dispersion
statistics, draw count, random seed, and upstream model identity.

### Measured selection

`CatalogueSelection` stores only atom-aligned pass probabilities. Its identity
includes the exact output predicate, flow/model view, blend response, shear,
draw depth, and independent random seed. Array and manifest replacement is
atomic.

### Frozen mocks

Likelihood-generated and image-generated mocks are distinct input kinds.
Likelihood mocks record the generating scene atom, density, shear transform,
selection, random streams, implementation identity, and output hashes. Image
mocks record their source catalogues, target mapping, injected shear, and
measurement-model hash; they do not carry a latent scene row or receive an
analytic blend injection because their pixels already contain blending.

`scripts/prepare_image_closure_mock.py` builds the latter from one declared
BlendEMU image leg. It rejects failed fits, reports dropped rows, and maps the
measurements into the flow checkpoint's exact target order.

## Exact and importance evaluation

The exact finite sum is the small-catalogue oracle. For a large prior,
`sbsi.catalogue_sampling` uses the defensive mixture

```text
q_i(j) = epsilon pi_j + (1 - epsilon) q_local,i(j).
```

Every sampled likelihood contribution retains the exact `pi_j / q_i(j)`
correction. The global term gives every positive-mass atom support and bounds
the importance ratio by `1 / epsilon`.

The `v1.1-infer` local proposal works in flow-predicted measured-output space:

1. select a broad nearest-candidate support;
2. rerank it with the cached flow uncertainty;
3. evaluate the exact initial-shear mass `pi P_det L(initial)` on that support;
4. mix the normalized local mass with the defensive full prior; and
5. draw once, then freeze atom IDs and proposal probabilities for the complete
   two-component stencil and one full-2D Newton update.

Candidate likelihoods evaluated at the initial point may be reused there, but
their cost and identity are reported separately. The retained
`distance_kernel` proposal is a validation fallback, not the production
default.

The expensive object numerator is sampled:

```text
Ahat_i(g) = (1/M) sum_m [pi_jm / q_i(jm)]
                  P_det(S_g z_jm)
                  p_flow(xhat_i - b_jm(g) | S_g z_jm).
```

The population normalization remains an exact atom sum because its detector
and selection terms are cached. Object processing is streamed; no
catalogue-sized object-by-atom tensor is retained.

Draw ladders are nested prefixes. Each object owns a deterministic random
stream with fixed-width records, so increasing the maximum draw count cannot
shift another object or an earlier prefix. Common draws are mandatory across
every finite-difference view.

### Stratified evaluation

`estimator_mode` selects how the atom sum is split. The `mixture` default is
the defensive proposal above. The `stratified` alternative does not blend at
all: it sums the candidate support exactly and spends every draw on the
complement,

```text
Ahat_i(g) = sum_{j in S_i} c_j(g)
          + (1/M) sum_t 1[j_t not in S_i] c_jt(g) / pi_jt
```

with `c_j(g) = pi_j P_det(S_g z_j) p_flow(xhat_i - b_j(g) | S_g z_j)` and
`j_t` drawn from the detected prior. The exact stratum contributes no
variance. The two forms are related: the mixture's `epsilon` component is
already a complement estimator with `epsilon M` draws, because `epsilon`
cancels in `pi/(epsilon pi)`. Stratification therefore changes the allocation,
not the estimand, and both remain unbiased for the numerator.

`draw_stratified` reads its atom from the uniform column `draw_adapted` uses
for its global component, so at one seed the retained complement draws are the
mixture's own global draws and the two estimators are paired by common random
numbers. It requires the production-prefix allocation and a retained full
ladder, because the mixture ESS and maximum-weight rules that drive adaptive
stopping do not define its draw budget. It is a screen: a run that selects it
records the deviation and is labelled `custom`, not `v1.1-infer`.

### Weight diagnostics

A retained ladder also records, per object and per rung, the zero-view weight
diagnostics: effective sample size, the largest single draw's share, the
estimator's relative standard error, and the generalized-Pareto tail index
`k`. They are written per object to `one_step_moments.npz` and as percentiles
to the `weight_diagnostics` block of `result.json`. They follow the retained
ladder because only there does every object share one prefix; adaptive
allocation gives each object its own draw count and no rung is comparable
across objects.

`k` is the decisive one. At `k >= 0.5` the weight variance is infinite, so the
estimator has no root-M rate and a `1/M` finite-draw bias correction does not
apply; at `k >= 1` the mean is infinite too. The reported threshold is
`min(1 - 1/log10(M), 0.7)`. Rows whose tail cannot be fitted — a weight vector
taking too few distinct values — are reported as undefined and counted, never
as light-tailed.

ESS fraction does not compare across `estimator_mode`. The stratified arm's
exact stratum carries no variance and consumes no draws, so its ESS fraction
is mechanically lower while its estimate can be much better. The relative
standard error is the quantity that compares, because it folds in the share of
the total that is sampled at all:

```text
relvar = (1/ESS - 1/M) (T / (E + T))^2
```

with `E` the exact stratum and `T` the sampled mean. For the mixture the
stratum is empty and this is the usual `sqrt(1/ESS - 1/M)`.

## Numerical estimate and partitioning

The `v1.1-infer` estimator is controlled by `configs/inference.json`; numerical
constants should not be copied into wrappers. It:

1. initializes at the configured statistic of the retained measurements;
2. builds the posterior-adapted defensive proposal at that point;
3. evaluates the complete two-component finite-difference stencil with the
   same atoms and probabilities;
4. sums the per-object score and full information, verifies positive-definite
   catalogue information, and solves for one Newton update; and
5. reports model and robust covariance estimates from the same retained
   per-object moments.

The output records score and information matrices, ESS, maximum-weight
concentration, local/global contribution, proposal reference and reuse,
selection normalization, flow-evaluation counts, runtime, and all provenance
needed to reproduce the partition.

Independent object partitions may be combined only by summing their sufficient
score/information and count terms under an identical effective configuration.
The combiner rejects duplicate or overlapping partitions and mismatched model,
scene, cache, selection, proposal, stencil, implementation, or random-stream
identity. It does not average already-solved shear estimates.

## Workflow

Prepare reusable inputs once:

```bash
python scripts/build_scene_prior.py --help
python scripts/build_catalogue_blend_response.py --help
python scripts/prepare_image_closure_mock.py --help
```

Run inference through `scripts/run_inference.py`, using
`configs/inference.json` and `configs/likelihood.json`, inside a scheduler
allocation. Use `jobs/job_inference.sh` as the local reference for environment
and resource setup. All catalogue, checkpoint, cache, mock, partition, and
output paths remain explicit user inputs.

Do not regenerate a mock when comparing draw counts, proposal seeds, or
finite-difference choices. Load the same frozen mock so the comparison changes
only the declared numerical choice.

## Validation

`scripts/run_catalogue_closure.py` compares the exact oracle and importance
sampler on a common frozen mock. The retained powered-exact,
powered-importance, and paired-nonzero programs exercise the local
score/information identities. They are validation entry points, not additional
production workflows.

At minimum, a validation panel checks:

- exact-versus-importance agreement where the exact sum is feasible;
- stability over nested draw rungs, candidate support, and independent proposal
  seeds;
- score centring and recovery of both positive and negative injections;
- observed-information versus score-variance consistency;
- finite-difference convergence and the full two-component cross terms;
- ESS, maximum-weight concentration, defensive-global contribution, and tail
  behavior;
- cache feature parity and exact row alignment; and
- reproducibility from a frozen mock and complete manifest.

`scripts/compare_estimator_arms.py` reads two or more common-random-number
paired arms of a screen. It refuses arms whose model, cache, scene, proposal,
mock, observation window, or implementation identity differs, and refuses any
`pipeline_config` difference the caller has not declared with
`--allow-differing`. It then reports each arm's ladder with paired rung-to-rung
errors, the paired difference between arms at every rung, and the weight
diagnostics above.

A likelihood-generated closure tests algebra and numerical sampling under the
declared density. It does not establish that models trained on one population
transfer to another. An image closure additionally tests measurement,
detection, usable-event, and response-model adequacy and must be labeled
separately.

## Current limitation

The bundled `v3.2-like` classifier has no ellipticity-dependent input. Its
spin-0 feature contract permits caching, but cannot represent image
detection-selection response. The image-stage order is therefore:

1. stabilize the uncut full-model sampler across draw rungs and proposal seeds;
2. reconcile detection and usable-measurement modeling; and
3. only then interpret realistic measured-output selections.

Passing a measured-cut likelihood closure does not by itself authorize a
selected-image closure claim.

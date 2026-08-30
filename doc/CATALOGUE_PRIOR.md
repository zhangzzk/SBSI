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

#### Conditioned deterministic-mixture tail proposal

`tilted_stratified` keeps the exact stratum above and replaces its inefficient
prior draw on `C_i = S_i^c` with a fixed deterministic multiple-importance
sampler. Let the cached diagonal-Gaussian proxy log likelihood be

```text
ell_proxy,i(j) = -sum_d log sigma_jd
                 - 0.5 sum_d ((x_i,d - mu_jd) / sigma_jd)^2.
```

The three unconditioned components are proportional to

```text
r_1(j)   proportional to pi_j Pdet_j exp(ell_proxy,i(j))
r_2(j)   proportional to pi_j Pdet_j exp(ell_proxy,i(j) / 2)
r_pi(j)  = pi_j.
```

Each component is normalized separately on `C_i`; tempering changes only the
approximate likelihood, never the `pi Pdet` base measure. Their balance
mixture is

```text
qbar_C(j) = (1/2) r_1,C(j) + (3/8) r_2,C(j) + (1/8) pi_C(j).
```

Every eight-draw cycle has sources `(r_1, r_2, r_1, pi_C, r_2, r_1,
r_2, r_1)`, hence exact `4:3:1` source counts. Regardless of its source, every
draw is divided by the same exact `qbar_C`. Thus

```text
Ahat_i(g) = sum_{j in S_i} c_j(g)
          + (1/M) sum_t c_jt(g) / qbar_C(j_t)
```

is unbiased whenever `M` is a complete number of cycles. Fixing the component
counts removes between-component allocation noise and cannot increase variance
relative to iid draws from the same `qbar_C`. Fixed `T=4` and detected-prior
bridge screens both worsened median relative SE without reducing the k-failure
count, so this three-component recipe remains the selected base sampler. The
`pi_C` component gives full catalogue support and bounds
`pi_j / qbar_C(j)` by eight times the complement normalizer.
The sampler uses blocked two-pass normalization and sampling, so its working
memory is bounded by the atom and proposal-object chunks rather than catalogue
size times the full observation count. Each atom slab is scored for a fixed
batch of 32 observations, which reuses the catalogue coefficients and replaces
per-object GPU scans with batched matrix operations. At the fixed 262,144-atom
slab, the validated four-component stress benchmark measured 2.436 GB total
CUDA allocation including the static whole-catalogue proxy tables; the
selected three-component recipe uses less. Both the 262,144-atom and 32-object
proposal chunks are recorded separately from the target-flow chunks in custom-
run provenance. Object-local random records make the result invariant to this
execution batching up to the documented last-bit GEMM/GEMV rounding; the
sampled balance probability is always assembled from the same normalized block
and within-block probabilities used by the sampler.

The recipe is exported as `TAIL_PROPOSAL_RECIPE` and copied verbatim into the
resolved pipeline provenance. It has no runner flags for temperatures or
mixture weights. `tilted_stratified` remains a `custom` validation arm until
held-out convergence, reliability, efficiency, and independent-seed gates are
met; it does not change the `v1.1-infer` default.
`scripts/benchmark_tail_proposal_batching.py` is the light synthetic execution
check; production timing must run it through the scheduler on the
proposal-development window or a catalogue-sized synthetic fixture.
On a 12.76M-atom, 32-object, `K=M=16,384` synthetic fixture, A40 job 16121928
reduced median DM4 proposal construction from 28.254 seconds at batch one to
2.793 seconds at batch 32 (10.11x), with identical sampled ids. The batching
implementation is recipe-generic; this timing and memory measurement are
specific to the four-component DM4 recipe.

#### Independent-pilot exact-stratum diagnostics

`--diagnose-complement-pilot-stratum` is a development diagnostic, not an
estimator mode or release change. It requires the fixed tail-proposal
diagnostic and cannot be combined with the frozen `exact-oracle/v1` protocol.
Measured cuts and selection caches are rejected so post-cut row positions can
never be relabelled as the frozen absolute IDs.
There are no command-line controls for its pilot size, score, promotion rule,
promotion count, proposal recipe, execution chunks, or random stream. The
dedicated `jobs/job_complement_pilot_stratum.sh` wrapper fixes the already
consumed development rows 25,000:25,512, `K=M=16,384`, prefilter 131,072, and
base proposal seed 8701. It explicitly retains the release
`mean_observed_shape` strategy, reproducing the archived DM3 development
centre `[0.013610377120038044, 0.002740148300964851]` from the full frozen
mock. Confirmation-window measurements therefore contribute only to that
already archived full-mock centre; no confirmation row receives candidate,
flow, pilot, or diagnostic evaluation.

For each object, the deterministic base stratum `S0` is constructed first. A
discarded 8,192-draw pilot then samples the selected DM3 proposal conditioned
on `S0` with seed `base_seed + 1,000,003` (1,008,704 in the fixed development
job). If `ell = log(c/qbar)` is the exact target-to-balance-proposal log ratio,
the explicit discovery score is

```text
2 ell + log(qbar) = log(c^2 / qbar).
```

Duplicate atom draws retain their maximum score. The active direct-v2 rule
ranks unique finite-score atoms by descending score and ascending atom ID,
then promotes the first 4,096 atoms beyond `S0`. If the pilot supplies fewer
than 4,096 unique finite-score atoms, the remaining slots are filled in
ascending active-atom order, excluding `S0` and the selected atoms. No
coordinate index or neighbour query is used. Pilot draws are selection data
only: they never enter an estimate, standard error, or Pareto-tail fit.

The archived clustered-v1 rule took 256 ranked anchors and filled the remaining
slots with standardized-coordinate neighbours. Its frozen primary run was
valid but failed every reliability and efficiency gate: only 390/466 of 512
objects had `k<0.5`/`k<0.7`, complement relative-SE median/p90 was
0.022191/0.078543, and mean/p90 variance-time scores were 1.478/1.612. It is
retained as a reproducible API option but is rejected as a proposal candidate.
The v1 result showed that anchors carried substantially more exact target mass
per slot than neighbours and that the coordinate query alone cost 46.087 s;
those observations fixed direct-v2 before its primary run.

Direct-v2 was subsequently run once under its frozen primary protocol. All
integrity gates passed, but it also failed every performance gate: only
397/460 objects had `k<0.5`/`k<0.7`, complement relative-SE median/p90 was
0.022873/0.077714, and mean/p90 variance-time scores were 1.919/1.842. Its
lower-ranked pilot atoms captured less exact mass than clustered neighbours
and worsened total relative error. It is therefore also rejected. The reserved
second pilot seed was not run and neither pilot strategy is an accepted
inference mode.

The pilot draw is bound to the complete DM3 component recipe, fixed
262,144-atom and 32-object proposal chunks, and a canonical per-object hash of
the exact `S0` exclusion. Promotion rejects a draw whose recorded recipe,
chunks, schedule, or exclusion hash differs from its supplied base stratum.

The final stratum `S` is reevaluated canonically with proposal probability one
and summed exactly, including promoted atoms outside the original prefilter.
Fresh prior-reference and DM3 complement streams then use `S` and the unchanged
production diagnostic seed `base_seed + 7,000,003` (7,008,704 in the fixed
job). Thus the production estimates are independent of the discarded pilot
conditional on the selected `S`, while retaining common random numbers between
the prior reference and DM3 arms. Outside-prefilter diagnostics explicitly
exclude every member of final `S`, so a promoted atom cannot be counted both
exactly and in the sampled outside mass.

Outputs record pilot and final-stratum flow counts and timings; the immutable
pilot/production recipes and seeds; recipe and run hashes; and per-object base,
promoted, anchor, neighbour, fallback, and outside-prefilter counts. Exact
base, promoted, anchor, neighbour, and fallback masses are separate table
columns. Each absolute object ID carries canonical hashes for its exclusion,
pilot atoms/probabilities/scores, base stratum, anchors, promotions, and final
stratum. Batch-shaped call hashes are retained separately as execution
provenance. The run manifest hashes the sorted `(absolute object ID, final
stratum hash)` pairs, not those batch-shaped arrays, so its scientific identity
is composable across sliced calls and independent of how already identical
per-object hashes are grouped. Numerical reproduction fixes the recipe's
262,144/32 proposal chunks; outer diagnostic partitions of at least 32 objects
must split on those 32-object boundaries, because changing the underlying
GEMM/reduction shape can move last bits in probabilities and scores. With the
pilot flag absent, existing table columns, metadata keys, flow counts, and
stochastic streams are unchanged.

The frozen direct-development protocol label is
`SBSI/complement-pilot-stratum-development/v2-direct`. Its expected identity is rows
25,000:25,512 at centre
`[0.013610377120038044, 0.002740148300964851]`, base/final stratum sizes
16,384/20,480, prefilter 131,072, `M=16,384`, base proposal seed 8701,
production diagnostic seed 7,008,704, and primary pilot seed 1,008,704. It also
pins the torch candidate backend, diagnostic/flow chunks 64/4,096, both DM3
proposal chunks 262,144/32, the complete tail and pilot recipes, pilot recipe
hash, hash contract, and SHA-256 identities of the frozen measurements, truth,
and likelihood-mock manifest. The job wrapper recomputes those three source
hashes before starting. The exact expected flow counts are 4,194,304 pilot,
10,485,760 final-stratum, and 98,566,144 total evaluations. A toy or partial
call is still valid, but its identity reports `matched=false` and its frozen
development gates fail; identity mismatch is not an API error.

The development summary preregisters three hard gate families. Integrity
requires the exact 512 integer-typed, int64-compatible absolute IDs and
protocol identity; 512 finite positive `stratum + tail` masses and tail
standard errors; nonnegative integer-typed count columns; base/promoted/final
counts 16,384/4,096/20,480 for every object; anchor count equal to
`min(finite-score count, 4096)`; fallback count
equal to the remaining promotion slots; zero coordinate-neighbour and query
counts; 16,384 retained tail draws; at least one finite-score pilot atom; and
internally consistent per-object hashes, execution-call coverage, table
final-stratum hashes, and run manifest. Tail reliability requires finite
Pareto `k` for all objects, at least 461 with `k < 0.5`, at least 507 with
`k < 0.7`, and complement relative-SE median/p90 no larger than
0.00882/0.0268. The p90 is NumPy's discrete `higher` quantile, inclusive at
the threshold.

Fair efficiency uses
`rT = tail_standard_error / abs(stratum_mass + tail_mass)`. The frozen archived
same-centre DM3 baseline has `rT` median 0.013214266498501442, `higher` p90
0.021110866788308195, mean `rT^2` 4.6256941067948046e-4, `higher` p90 `rT^2`
4.45668696553694e-4, and method time `T0=91.04022970038932 s`. Pilot method
time `T` includes candidate query, prefilter flow, pilot proposal draw and
flow, promotion, final exact-stratum flow, tail proposal draw and flow, and
reduction. It excludes the prior-reference draw and flow. Both
`(T/T0) mean(rT^2)/baseline_mean` and
`(T/T0) p90_higher(rT^2)/baseline_p90` must be at most 0.90. The summary also
reports `T/T0` and candidate total elapsed divided by the baseline total
elapsed 103.91162065404933 s; total elapsed is context, not substituted for
the fair method time. Missing, nonfinite, negative, or zero total method time
fails closed. Overall `gates.passed` requires all integrity,
tail-reliability, and efficiency gates.

Pilot seed offset 2,000,003 is separately reserved before seeing the primary
outcome. If and only if the primary frozen run passes every gate, the
predeclared replication repeats the identical protocol with pilot seed
2,008,704 and no other change. There is intentionally no replication CLI or
seed-2 job yet. Promotion into the production estimator remains out of scope;
an exact-confirmation protocol would require an intentional version after
selection and must not reuse the frozen exact-oracle-v1 rows.

#### Spatially stratified pilot/Neyman centre diagnostic

`--diagnose-complement-spatial-neyman` is a centre-only development
diagnostic, not an estimator mode or release change. It tests whether changing
coverage, rather than enlarging another object-specific exact support, repairs
the heavy conditioned-DM3 tail. The protocol fixes rows 25,000:25,512, the
archived mean-observed-shape centre
`[0.013610377120038044, 0.002740148300964851]`, Torch candidate search,
`K=16,384`, prefilter 131,072, DM3 production size 16,384, proposal seed 8701,
pilot seed 1,008,704, and production seed 7,008,704. It requires the frozen
mock hashes and the independent exact-oracle selection manifest. Measured
cuts, selection caches, learned retrieval, the earlier pilot-promotion path,
and the ordinary tail-proposal diagnostic are mutually excluded. The
diagnostic was disabled until the outcome-blind partition described below had
three independently reviewed hashes; those semantic, cache-byte, and
manifest-byte identities are now embedded in `SPATIAL_NEYMAN_PROTOCOL`, and
science fails closed on any drift.

The global active support is partitioned once into 256 cells using only
catalogue-side quantities. Starting from all active atoms, eight binary levels
split axes `(0,1,2,3,0,1,2,3)` in the four float64 standardized proposal
coordinates. Within each node atoms are stably ordered by coordinate and then
atom ID. The chosen cut minimizes the distance from one half of the node's
detected-prior mass, subject to a 45--55% count guard; ties choose the smaller
cut, and a zero-detected-mass node uses its count midpoint. Left/right binary
paths define cell IDs. The count guard ensures every audited science cell has
more than 16,384 and no more than 262,144 active atoms, so removing an entire
exact stratum cannot empty a cell and one cell fits the frozen mapping slab.

Partition construction is a separate Slurm audit that returns before loading
a mock, querying candidates, or evaluating the target flow. The audit writes
the compact assignment/count/mass cache plus a manifest containing the recipe,
input-array, assignment, count, detected-mass, and prior-mass identities. A
science run separately verifies the semantic `partition_sha256`, exact cache
bytes, exact manifest bytes, embedded identities, live proposal inputs, and
cell execution bounds. The audit wrapper and science wrapper both require a
clean detached commit and refuse overwrites. Partition building is not charged
to science, while actual science-side cache/manifest loading and semantic
verification are charged.

For object `i`, let `S_i` be the deterministic 16,384-atom exact support and
let `H_h` be cell `h`. On `H_h \ S_i`, each of the three DM3 source
components is normalized separately. The balance law is

```text
q_ih = (1/2) r_ih,t1 + (3/8) r_ih,t2 + (1/8) r_ih,prior.
```

The prior component gives full complement support. A proxy component whose
cell normalizer is exactly zero falls back to the cell prior; NaN or positive
infinity is an error. Pilot and production reservoirs reuse the same exact
cell normalizers and `q_ih` values but use independent counter-keyed random
domains. The discarded pilot has two complete DM3 cycles per cell, hence
4,096 draws total with fixed source counts 2,048/1,536/512. The production
reservoir has 64 cycles per cell, hence 131,072 mapped draws; only selected
whole-cycle prefixes are target-flow evaluated. Normalization and mapping of
both reservoirs occur in one cell pass, without a dense cell-by-catalogue or
object-by-catalogue probability table.

If `Y=c/q_ih`, the discarded pilot estimates the allocation score

```text
V_ih = sum_l alpha_l sample_variance(Y_ihl),
alpha = (1/2, 3/8, 1/8).
```

Production starts at two cycles in every cell and assigns exactly 1,536
additional cycles, one at a time, to the uncapped cell maximizing
`V_ih / (m_ih (m_ih + 1))`; cell-ID order breaks ties. Each cell is bounded to
2--64 cycles and every object has exactly 2,048 cycles, or 16,384 selected
draws. An all-zero pilot score has a deterministic eight-cycle-per-cell
fallback, but any such fallback fails the science integrity gate. Pilot draws
never enter the estimate, standard error, or Pareto fit.

The centre estimator is

```text
Z_hat_i = sum_[j in S_i] c_ij
          + sum_h mean_[J in production prefix h] c_iJ / q_ih(J).
```

It is unbiased conditional on the fixed exact support, partition, pilot, and
allocation because the independent production mean in every cell integrates
that cell's complement. With fixed source counts `n_ihl` and total cell count
`n_ih`, its conditional variance estimate is

```text
Var_hat(complement_i)
  = sum_h sum_l n_ihl * sample_variance(Y_ihl) / n_ih^2.
```

The equivalent pooled vector for the Pareto diagnostic is
`(M/n_ih) Y_ih`, where `M=16,384`; raw `Y_ih` values from unequally allocated
cells are not pooled. An independent reduction recomputes the sum of cell
means and fixed-source variance and must agree with the sampler core.

The fixed execution uses eight-object proposal batches, 262,144-atom spatial
slabs, and 4,096-atom target-flow slabs. Exact-support, pilot, selected
production, and exact-oracle target-flow counts are respectively
8,388,608, 2,097,152, 8,388,608, and 816,703,360; total diagnostic target-flow
count is 835,577,728. Method time excludes the exact oracle but includes
proposal initialization, science-side partition load/verification, candidate
query, exact-support flow, all reservoir normalization and mapping including
unused production entries, pilot flow, allocation, selected production flow,
and reduction.

Integrity additionally requires exact absolute IDs and protocol identity,
the three frozen partition identities and cell bounds, unique active exact
supports, exact fixed source/cell/draw counts, distinct RNG streams, zero
support leakage, zero all-zero allocation fallbacks, whole-cycle bounded
allocations, matching support/exclusion hashes, exact core call provenance,
recomputed run manifest, exact flow counts, and finite positive masses and
standard errors. Per-object output records the number of proxy-component cells
that used the exact prior fallback; fallback is valid, but it may not be
silent. Reliability retains the preregistered learned-centre bars:
at least 461/512 finite `k<0.5`, at least 507/512 `k<0.7`, no `k>=1`, and
complement relative-SE median/higher-p90 at most 0.00882/0.0268.

The same outcome-independent 64-object exact subset requires maximum exact-
support cross-check error `1e-5`, median/higher-p90 absolute total-log-evidence
error 0.01/0.05, aggregate `|z|<=2.5`, exact-error z RMS in 0.8--1.2, and at
least 58/64 `|z|<=1.96`. Against the frozen DM3 baseline mean/higher-p90
squared total relative errors
`4.6256941067948046e-4`/`4.45668696553694e-4` and method time
`91.04022970038932 s`, both variance-time ratios must be at most 0.90. All
integrity, reliability, exact-calibration, and efficiency gates must pass
before any stencil or inference integration is authorized.

#### Frozen exact-oracle confirmation

`--diagnose-complement-exact-oracle` is a diagnostic-only extension of the
tail-proposal screen. It requires both `--diagnose-complement` and
`--diagnose-complement-tail-proposal`; it never changes estimator mode. The
declared diagnostic universe must contain exactly 512 consecutive absolute
object IDs. There are no oracle seed, count, or chunk command-line knobs. A
measured cut or selection cache is rejected before output creation: otherwise
post-cut row positions could be mistaken for the frozen absolute object IDs. A
run is labelled as the predeclared confirmation only when it also matches rows
25,512:26,024, centre `(0,0)`, `K=16,384`, prefilter 131,072, `M=131,072`,
proposal seed 9701, and the derived diagnostic seed 7,009,704. The execution
identity additionally fixes the Torch candidate backend, 64-object diagnostic
and 4,096-atom target-flow chunks, 262,144-atom and 32-object tail-proposal
chunks, and 65,536-atom and four-object exact chunks. Development or forensic
runs may compute the same metrics but fail the protocol-identity gate. The
identity pins the complete `asdict(TAIL_PROPOSAL_RECIPE)` DM3 recipe, selection
key, and fresh-window selection manifest as well as the frozen mock's
measurements, truth, and manifest SHA-256 values; the runner independently
rehashes those files before the diagnostic. A future selected proposal must
therefore version and deliberately update this protocol rather than silently
inheriting the v1 confirmation label.

The fixed label is `SBSI/complement-exact-oracle/v1`, whose SHA-256 digest is
`f02f535eab874dd6f9f6184d4040283493266074903fc6c20160b4ae88b5993e`.
For every absolute object ID `i`, selection scores
`SHA256(key_bytes || uint64_be(i))`; the 64 smallest scores are retained, then
their IDs are put back in ascending order. The manifest is SHA-256 over those
64 sorted big-endian uint64 values. Its frozen hashes are
`6abcc361aec9ada6010f2a91a68aa70440904873518182445c2a712ae69bad6a`
for the development universe 25,000:25,512 and
`fc44a632ed4758440a3e241ffeab9fba087f58397f114cfbc38aecc7feedcb8b`
for the untouched confirmation universe 25,512:26,024. Selection therefore
uses neither observations nor proposal outcomes.

For each selected object the oracle visits `proposal.active_indices` exactly
once in 65,536-atom blocks and four-object subchunks. It calls
`log_importance_weights_tensor` with `q=1`, masks the exact stratum `S`
directly, and accumulates separate `S` and `S^c` log-sum-exp reductions in
float64. The streamed `S` result must reproduce the already evaluated
prefilter-stratum sum. Only the selected 64 table rows receive exact mass,
exact z, and total-evidence-difference values; the other 448 rows carry nulls.
The output records the selected IDs and manifest, active-support hash, chunks,
flow count, code/wrapper hashes, and the following predeclared gates:

- integrity: an exact match to the frozen confirmation protocol, exactly 64
  selected objects, every active atom visited once per selected object, finite
  positive exact masses and proposal standard errors, and maximum relative
  stratum cross-check error at most `1e-5`;
- accuracy: median absolute tail-minus-exact change in log total evidence at
  most `0.01`, with the higher-order-statistic p90 at most `0.05`;
- signed aggregate calibration:
  `abs(sum(DM - exact) / sqrt(sum(SE^2))) <= 2.5`;
- per-object calibration: exact-z RMS in `[0.8, 1.2]` and at least 58 of 64
  exact z values within `[-1.96, 1.96]`; and
- all 512 objects, unchanged from the proposal screen: no undefined Pareto k,
  at least 90% with `k < 0.5`, at least 99% with `k < 0.7`, and complement
  relative-SE median/p90 (higher method) at most `0.00882 / 0.0268`.

The exact subset cannot relax an all-object proposal gate. The dedicated
`jobs/job_complement_exact_oracle.sh` wrapper fixes the fresh universe and
checks its selection manifest before starting the production-depth screen. Its
tag identifies the selected three-component `T1:T2:prior = 4:3:1` DM3 recipe,
and it explicitly requests the fixed zero centre rather than inheriting the
release's data-derived optimizer start.

### Weight diagnostics

A retained ladder also records, per object and per rung, the zero-view weight
diagnostics: effective sample size, the largest single draw's share, the
estimator's relative standard error, and the generalized-Pareto tail index
`k`. They are written per object to `one_step_moments.npz` and as percentiles
to the `weight_diagnostics` block of `result.json`. They follow the retained
ladder because only there does every object share one prefix; adaptive
allocation gives each object its own draw count and no rung is comparable
across objects.

For an unbounded continuous tail, `k >= 0.5` corresponds to infinite variance
and `k >= 1` to an infinite mean. SBSI instead sums a finite catalogue with a
full-support defensive component, so every exact importance moment is finite.
Here `k` is an operational, pre-asymptotic diagnostic: a high value says the
available draw ladder behaves like a heavy tail and does not yet justify a
root-M or `1/M` extrapolation. The reported threshold is
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

For the deterministic mixture, pooled ESS and Pareto-k remain concentration
diagnostics but pooled ESS is not a variance estimator: its between-component
spread is not random. The reported relative standard error instead combines
the unbiased within-component sample variances under the fixed `4:3:1`
allocation. Per-rung component counts and per-object pre-conditioning
complement masses are stored with the one-step moments and summarized in
`result.json`.

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

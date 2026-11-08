# Likelihood and inference

## Current identities

This checkout has one inference workflow with independent numerical and likelihood
identities. Later inference work on another branch is recorded under
[cross-worktree tests](#later-cross-worktree-inference-tests):

| Role | Identity | Configuration |
| --- | --- | --- |
| Default estimator | `v1.1-infer` | `configs/inference.json` |
| Tilted-stratified estimator | `v1.2-infer` | `configs/inference_v1_2.json` |
| Default likelihood | `v3.2-like` | `configs/likelihood.json` |
| Retained development model set | `V3.9-like` | `configs/models_v3_9_like.json` (artifact manifest only) |
| Historical development model set | `V3.6-like` | `configs/models_v3_6_like.json` (artifact manifest only) |
| Historical model set | `V3.5-like` | `configs/likelihood_v3_5_like.json` |

`V3.5-like` is not the default and has not demonstrated 0.3% calibration. It
names lambda10 epoch154, trial9 `R_blend`, and the three-seed nine-input
coherent-`U` probability ensemble.

V3.6-like names the single common-descent flow, disk-response emulator and
single smooth-crowding classifier retained on 2026-09-20. The investigation
was stopped by owner decision. Its manifest is deliberately not a likelihood
configuration: selecting that manifest alone does not select the disk
composition or state-dependent classifier. See
[runtime boundary](JOINT_CALIBRATION.md#runtime-and-inference-boundary).

The subsequently requested uncut500k/16384-draw run uses the dedicated disk
driver described below. `configs/likelihood_v3_6_like.json` pins its cuts and
artifacts; its **preparation-only** flag intentionally keeps the legacy
additive driver from accepting it. All disk production partitions completed,
but the combined curvature check failed. The investigation is closed; no valid final shear estimate was produced.

### V3.9-like boundary

The owner-retained model is now F8+C8+disk. See
[the model contract](V39_MODEL.md#loading-and-inference-boundary).
The disk driver below retains V3.6-specific features and artifact pins.
V3.9 requires newly validated physical-crowding preparation; no old disk
cache or prior feature store is implicitly compatible. All prepared caches
now require exact scientific and implementation identity; the one-off
historical LRU patch exception was retired during cleanup.

### V3.6-like disk driver

`scripts/run_disk_inference.py` now supplies disk-specific prepare/assemble/run
stages around the shared `v1.2-infer` estimator. Its preparation-only config
remains rejected by the legacy additive driver. The dedicated path validates
the three model hashes, uncut subset receipts, pairing geometry, observing
conditions, input hashes and implementation identity. It persists state-major
classifier probabilities, globally scaled proposal coordinates and CRN
population normalization; pair features remain sharded and memory-mapped.

The16384-atom/128-observation smoke pilot and full24million-atom/256-observation
compiled pilot passed. A two-entry response cache fixed exact/complement
eviction; the historical replay preserved every pilot array bitwise. The500k production run
completed all20observation partitions. The strict combiner failed because the
combined information is not positive definite at any draw rung, including16k;
there is no valid final shear estimate. The negative curvature is concentrated
in a small bright-object tail. Historical exact-sum and paired-draw probes are archived with their findings. This does not establish prior-size convergence or
shear calibration. See [production curvature audit](../archive/research-2026-09-25/snapshots/doc/WORKLOG.md#2026-09-21--diagnose-failed-v36-production-curvature).

The historical100k rerun remained gated and was not submitted. The
[end-to-end review](../archive/research-2026-09-25/doc/V36_INFERENCE_REVIEW.md#findings-by-layer) records the
likelihood, population-support, finite-prior, QMC, sampling, derivative and
solver limitations; the [launch checks](../archive/research-2026-09-25/doc/V36_INFERENCE_REVIEW.md#predeclared-acceptance-checks)
separate proposal development from acceptance and prohibit empirical shear
corrections or population changes to obtain a favorable result.

### Later cross-worktree inference tests

The September 21–22 tests on `worktree-proposal-coord-compare` were not
integrated into this checkout. That branch makes `v1.3-infer` its default:
the tilted-stratified estimator with K=1024, 16384 complement draws, and a
median (50th-percentile) proposal-scatter floor, fractional on flux. Its
[work log](../.claude/worktrees/proposal-coord-compare/doc/WORKLOG.md)
records subsequent prior experiments. The saved disk runs identify themselves
as `custom`, based on `v1.2-infer-16k`, with V3.6-like model hashes; the branch
default name must not replace their recorded provenance.

A 20,231,221-atom prior includes a 20m uniform base and every remaining
source atom with true r<20, with inverse-inclusion weights. This enriches
prior resolution without imposing a truth cut on the observed sample.
Two disjoint 10000-observation windows both have positive-definite summed
information. Re-reduction of their hash-verified saved moments on September25
gives:

| Observations | Estimated g1 | Robust standard error |
| --- | ---: | ---: |
| Rows 0–9999, job16631250 | 0.03482388 | 0.01470535 |
| Rows 10000–19999, job16631862 | 0.05117895 | 0.08590561 |
| Pooled 20000 | 0.03768952 | 0.01724200 |

Injected g1 is 0.02. The estimator is `g_center + solve(I, score)`;
earlier branch entries reported the step alone (pooled 0.02377725), omitting
the common starting g1=0.01391227. The later branch entry corrects this.
These are noisy one-step estimates, not accepted calibration results.
About 31% of rows have estimated Pareto k>0.7; positive information alone
does not demonstrate integration or prior convergence.

A subsequent truth-r<27 prior with 10,237,320 atoms (jobs16637857/16637858)
fails positive definiteness when the two windows are pooled. It changes both
support and atom count, so those effects cannot be separated. All these
tests use V3.6-like models and retain measured radius>0.6 arcsec/magnitude<25.8
selection. They do not establish V3.9 inference integration or robust 0.3%
accuracy. Their result JSON and moment arrays are preserved under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/`.

The September25 [sampled-atom maps](figures/inference_atoms_20260925/README.md#results)
replay this proposal on the original diagnostic rows142230/3563/409188 using
the 20.23m prior and a hash-matched runtime. Full-prior centre-node scans show
99.9247%,9.4684%,30.6568% posterior mass reached, respectively; the latter two
rows still miss dominant atoms. This does not establish derivative accuracy.

## Historical mock recovery and the V3.9 one-step limitation

The September26 owner-requested work-log review recovered a known estimator
failure mode that must accompany interpretation of the V3.9 model-mock500k
result: a single Newton update from mean measured shape need not locate the
maximum. This is already explained in [the local-expansion boundary](MATH.md).

The August22 profiles (archived WORKLOG_FULL, cont.204) recovered injected
axis-aligned shears0.02 and0.05 within0.31 profile-curvature standard errors
in all four10000-object tests. On those same objects the zero-centred one-step
estimator overshot or encountered negative information. These were older
shape-only likelihood tests, without measured cuts or blend response, and
their relative statistical precision was1.2–3.7%, not0.3%.

The August30 full25000-object Newton chain (cont.342) moved
g1=0.022960 ->0.01906560 ->0.01958117 for injection0.02, using previous
estimates as subsequent centres. The final robust conditional SE was0.00061195.
The saved `fixedpoint_chain1_k1024/result.json` and
`fixedpoint_chain2_k1024/result.json` under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_stratified_screen_v1/results/`
were reread on September26 and reproduce those numbers. K1024/M8192 used
Gaussian reranking with131072 prefilter candidates. The separate saved
`inference_hybrid_chain_v1/hybrid_n20k_mag22_5pass/chain.json` confirms the
20000-object bright-recompute/linear-carry chain settled at
g1=0.01953533 ±0.00071477 after five passes; its final three steps moved
g1 by at most8.2e-7. That hybrid approximation is not validated for V3.9.

A later September26 audit found two material omissions from that initial
historical recovery. The August23 safeguarded optimizer (cont.205) capped steps
and accepted them only when the fixed-draw likelihood increased; its four
10000-object results agreed with their injected components within0.34 curvature
standard errors. More directly relevant, the September1 complete-likelihood
100k run (cont.369–370) used **whole-prior Gaussian top-K**, normalization-aware
measured-magnitude<22 carry, and **BFGS score-root updates** after the first
observed-information step. Its saved chain is
0.03479970 ->0.02216762 ->0.01777542 ->0.01997634 ->0.01992642,
with final conditional SE0.00041377. This older likelihood used different models
and selection (including radius0.75arcsec); it is evidence for the numerical
procedure, not a current0.3% result. Plain unsafeguarded Newton with a131072-atom
location prefilter does not reproduce this later successful procedure.

These results demonstrate recovery consistent with truth and sensitivity to
recentering in the older likelihood. They do not prove zero bias, certify
0.3%, or establish that recentering alone fixes the current likelihood.
The old uncut V1 shape-flow test at8million objects reported
+0.034% ±0.252% (cont.174); its selected-sample claims were subsequently
corrected for population-normalizer uncertainty (cont.175). Its shape cut
|e|<0.6 is not the current measured radius>0.6arcsec cut.

The completed V3.9 mock500k driver still evaluates one stencil at raw mean
shape(0.01303596,-0.00261937) and returns centre+solve(information,score).
Its +4.456982% ±2.186859pp relative g1 error is therefore a one-step result,
not a converged likelihood maximum. Its13.492342pp change between8192 and
16384 draws independently shows unresolved draw sensitivity. Both issues
must be separated before attributing the result to models or sampling alone.
Unlike those older mocks, it uses F8/C8, disk/Möbius response, the expanded
weighted prior, and measured radius/magnitude selection.

The appropriate next diagnostic is a fixed subset of the existing V3.9 mock,
with paired draw-budget comparisons and full recentered likelihood passes or
a local two-component profile. Recompute the selection-normalization stencil
at every new centre, retain common random numbers, and use previous estimates
rather than injected truth to choose centres. Check normalization accuracy
separately. Stop on score/step and numerical stability, never closeness to
truth. Historical catalogue-split errors must not be pasted onto the current
same-finite-prior mock. No new experiment or archived launcher was started
as part of this review.

The owner subsequently authorized the [20k iterated diagnostic](V39_RECENTER_20K.md#purpose-and-prior-revision)
and requested removing the new bright extension. That study generates a fresh
matched-prior mock and tests both retrieval arms at shared iterated centres;
it does not reuse expanded-prior observations with a reduced inference prior.
It was stopped after two full passes when the second Newton correction grew.
The [saved-moment audit](V39_RECENTER_20K.md#historical-procedure-correction-and-saved-moment-audit)
quantifies the bright-object nonlinearity and a historical BFGS alternative;
neither that alternative's next centre nor the stopped Newton chain is a
converged inference result.
The subsequent owner-requested [five bright-only BFGS iterations](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point)
start from the first full-pass estimate, recompute measured magnitude<20,
and carry all other observations from that first full evaluation. This is
a separate approximation study, not a changed default or an accuracy claim.
The owner then requested the [same procedure on20k actual ConstGold measurements](V39_ACTUAL20K_BFGS.md#request-and-sample).
That run computes its own full first pass before five fresh bright updates;
it retains the measurement cuts, V3.9 models, restored prior and sampler.
After that actual run stopped at its first indefinite information matrix,
the owner requested [forward mocks from its exact input identities](V39_MATCHED_INPUT_MOCK.md#owner-request-and-generation).
This comparison fixes the failing expansion point and reports regenerated
measurements alongside actual measurements of the same retained identities.

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
replacement. No minus-leg detection/selection is applied. Historical defaults
retain their original usability rules. Explicit `--no-truth-cuts` and
`--no-truth-radius-cut` flags remove the corresponding truth bounds. V3.6-like
uses its classifier's finite, positive-radius/flux, unit-disk usability contract
and strict measured bounds; selected true-r>=26 objects are reported as outside
its training parent. The resulting image-input manifest pins source hashes, row identities,
the flow checkpoint, selection, and sample seed. This is input preparation
from existing measurements; it does not simulate images or measure shapes.

To expand the active prior while keeping the same complete source scenes,
`scripts/prepare_prior_domain_cache.py` validates full-shard truth features and
row identities, selects the declared primary domain, and recomputes R_blend
with the full neighbour graphs. Its compact output preserves only verified
deterministic zero-shear features. `build_model_cache_from_zero_view.py` then
evaluates fresh classifier probabilities and flow-QMC coordinates on that
support. The [0.37/0.5-arcsec ConstGold rerun](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_RE037_SIZE050.md#authorized-setup)
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
See the [500k ConstGold process review](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_500K_REVIEW.md#what-the-original-uncertainty-computes)
for case resampling and the [numerical controls](../archive/research-2026-09-20/doc/INFERENCE_CONSTGOLD_500K_REVIEW.md#numerical-convergence-and-tail-diagnostics)
for limitations of that run.

## V3.5-like status and acceptance boundary

The V3.5-like cached mechanism screen on cases120--139 gave
`m=-0.372 +/- 0.223%`. Its truth-population and measurement-residual
contributions were `-0.549pp` and `+0.177pp`, so the central value depended on
opposing errors and was not accepted. The actual-`U` oracle across cases40--139
gave `m=-0.129 +/- 0.125%`; its approximate 95% interval extending slightly
beyond `-0.3%` is acceptable for routing work to the classifier, not for final
promotion.

For this historical V3.5-like release, a final claim requires a frozen candidate
evaluated on fresh multi-axis image simulations with its fixed
0.75-arcsec/no-`|e|` sample, numerical-QMC control, and componentwise
anti-cancellation gates. The separate joint-calibration investigation used the
owner's 2026-09-18 primary cut: measured FLUX_RADIUS >0.60 arcsec and
MAG_AUTO<25.8, with strict true r<26. Its population is specified in
[the joint-calibration protocol](JOINT_CALIBRATION.md#goal-and-retained-population);
this clarification does not change the configured likelihood's selection. No response offset, `R_blend`
rescaling, temperature fit, threshold tuning, or result pasted from a reused
endpoint is allowed.

Detailed historical derivations and experiment branches are preserved in
`archive/research-2026-09-14/doc/INFERENCE_FULL.md` and the archived work log.

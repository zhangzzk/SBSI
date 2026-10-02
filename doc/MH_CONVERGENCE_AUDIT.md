# Continuous-MH convergence audit

## Finding — October 2, 2026

The completed bright test has two separate numerical failures: incomplete
posterior exploration at the reference shear, and inadequate importance overlap
when reusing that posterior farther away in shear. A reporting interpolation
bug additionally made some intervals inconsistent with their fitted peaks.
The interval bug is fixed; neither scientific failure is fixed by this audit.

There is direct evidence for curved morphology/shape geometry that ordinary
Gaussian random walks handle poorly. For observation80, a conditional straight
path drops52.27 log-density units below its lower endpoint; a curved path
between the same endpoints has no sampled drop below that endpoint. This is a
specific route around the apparent barrier, not proof of global connectivity
or posterior convergence. Simply increasing chain length or relaxing Rhat is
not the preferred next experiment.

No acceptance, cache, or coordinate-Jacobian defect was found in the inspected
paths. Exact final-state replay passes. These checks do not prove the entire
implementation or scientific model correct. V3.9 artifacts, prior, observations,
source runs, seeds and inference defaults remain unchanged.

## Scope and conventions

This audit reads the [completed bright test](BRIGHT_MH_TEST.md#completed-result--october-2-2026):
128 measured-selected observations below20 and128 below22, with18 shared rows
and238 distinct galaxies. These are overlapping cohorts, not disjoint magnitude
bins. Selection uses measured properties; failed objects remain included.
Two independent replicas each have four chains,32768 steps and8192 warmup.
The within-replica block schedule is shared; proposal and acceptance randomness
is per chain. Independent replicas retain separate seeds.

The target is the fixed GalSBI continuous prior with F8/C8/original disk
likelihood at(g1,g2)=(.02,0). Fits are conditional in g1 with g2=0. These are
model-closure diagnostics, not FS2 recovery or a calibration-bias measurement.
See [conventions](CONVENTIONS.md) for shape/response definitions and population
cut order. No truth cuts, empirical offsets or altered likelihood are used.

Full-chain statistics use all238 objects. Geometry diagnostics use ten objects
selected after seeing the original failures: IDs177,167,80,184,83,67,42,79,225,209.
The selection covers local-ratio Rhat quantiles in measured<20 and20–22, plus
large density spread and cross-block correlation. Four conditional path probes
use177,80,225,209. These panels are exploratory, not independent validation.

## The local disagreement is scientifically relevant

Full convergence checks pass0/128 in both cohorts. Both local shear-ratio
checks pass35/128 below20 and10/128 below22. Their median Rhats are approximately
1.027/1.023 below20 and1.076/1.072 below22 for the .018/.022 nodes. Thus failure
is not confined to latent variables irrelevant to shear.

For a reference posterior pi0(theta) proportional to p(theta)K(theta,g0),

```text
I(g)/I(g0) = E_pi0[K(theta,g)/K(theta,g0)].
```

At .018 and .022, all128 objects in each cohort have raw importance ESS>400
and Pareto k<.7. Healthy observed weights do not establish exploration, but they
show that local chain disagreement cannot be dismissed as a wide-shear weight
problem. The actual shear-sensitive expectations differ across chains.

Rhat<1.01 is a diagnostic screen, not a rule to remove observations from the
population. Keep failures and route them to more reliable integration; dropping
objects by convergence could introduce additional selection. A passing scalar
Rhat never proves that all modes have been found. See the
[rank-normalized Rhat reference](https://arxiv.org/abs/1903.08008).

## Why 30 percent acceptance does not imply useful movement

Median morphology acceptance is .301 below20 and .311 below22; shape acceptance
is .302/.319. The sampler successfully accepts local steps while exploring some
important directions too slowly. Among the80 panel endpoints, the smallest
proposal-to-explored morphology covariance eigenvalue has median .0155 and
10th percentile .000404. In standard-deviation units those are12.45% and2.01%
of the explored width per proposed morphology move. Some are only1.14%.

The covariance denominator is the pooled explored distribution, not a converged
posterior reference: it can omit modes or include drift. Nonetheless the size
mismatch, differing chain traces and conditional path tests provide mutually
supporting evidence. At the full-population level, between-chain mean differences
account for median13.6%/17.1% of explored magnitude variance and8.9%/14.2% of
log-radius variance, for<20/<22 respectively.

The original morphology proposal couples magnitude, log radius and Sersic logit;
shape is a separate block. Covariance adapts per chain using the latest1024
warmup states every128 steps, with5% baseline-diagonal shrinkage. Scale adaptation
targets .3 acceptance and freezes after warmup. Local covariance can describe a
small segment of a curved distribution without representing the whole route.
This mechanism is consistent with the evidence; the audit has not isolated the
adaptation window as the sole cause.

## Direct conditional-path experiment

Hold redshift, SED and environment fixed to replica0/chain0 for each diagnostic
object. At17 Sersic-logit knots spanning final-chain endpoints, optimize magnitude,
log radius, distortion logit and phase with bounded differential evolution:
population16,200 iterations, seed2026100602. Bounds surround explored states and
are only optimizer search bounds, not changes to prior support. Evaluate the
unchanged prior plus likelihood on129 points along the interpolated optimized
path and along the straight line joining its same two endpoints.

| Observation | Straight-path drop below lower endpoint | Curved-path drop |
| --- | ---: | ---: |
| 177 | 3.538 | 0 |
| 80 | 52.270 | 0 |
| 225 | 0.199 | 0 |
| 209 | 0 | 0 |

For80, the maximum curved-minus-straight difference is71.46 log-density units.
The accompanying trace plot shows a curved Sersic–radius relation with chains
occupying different portions. The curve makes a geometry-aware proposal worth
trying; it does not provide posterior samples, relative mode masses, or an
estimate of recovery. All barriers are evaluated on finite grids, and the fixed
nuisance variables/environment limit what connectivity can be inferred.

The earlier full-ten-coordinate straight-line test is a different conditional
experiment. Its barriers must not be compared numerically to this table as if
both shared endpoints and fixed variables.

## Simply widening a Gaussian is not a general repair

At each of80 final states, evaluate128 proposals using the exact unchanged
target. Compare the original morphology Gaussian with pooled explored3D and5D
covariances at .1,.3 and1 times the conventional2.38/sqrt(d) scaling. These are
one-step diagnostics at nonconverged states, not new stationary chains.

Averaged across the panel, original expected acceptance is .311 and standardized
expected squared jump distance .0914. Full-size3D/5D alternatives have only
.052/.040 acceptance and jump distances .147/.147. Improvements are uneven.
For difficult observation80, the original jump distance is about .004; a
full-size3D proposal falls close to zero, and5D is about .001. Some other
observations benefit. One enlarged Gaussian cannot reliably follow the curved
paths; the existing failed joint-block/pooled/stretch experiments should not
be presented as untried solutions.

Median maximum within-chain morphology–shape correlation is only .309/.294
in<20/<22, although the upper tail is larger. Strong linear correlation between
blocks is therefore not a universal explanation; low linear correlation also
cannot exclude nonlinear dependence.

## Other hypotheses checked

- **Stale cache or mismatched accepted state:** reconstruct80 retained final
  scenes including their environments. Maximum prior and kernel differences
  are exactly0; compressed-summary error is2.84e-14. This checks the audited
  states, not every transition in every run.
- **Target-density and MH correction:** reviewed native red/blue mixture,
  transformed-coordinate Jacobian, wrapped phase, SED-jump Hastings correction,
  prior-reversible partial/full Poisson environment refresh, warmup-only
  adaptation and accepted-state copying. Existing tests include Jacobian
  finite differences, mixture/Hastings checks, Poisson refresh stationarity,
  ragged-scene ownership and Gaussian MH closure. No defect was identified.
- **All chains permanently retain their initial Sersic levels:** median
  initial-versus-final chain-mean Sersic rank correlation is .095 below20 and
  -.048 below22. Simple universal initialization locking is not supported.
  Initialization can still affect individual failures.
- **Environment never changes:** full-refresh acceptance has median .635/.425,
  though individual chains can have zero. Persistent-neighbour sampling is not
  universally frozen; difficult coupled target/environment cases remain possible.
- **Redshift/SED moves are useless errors:** these moves intentionally leave
  compressed likelihood summaries unchanged while changing the joint prior.
  They consume25% of the block schedule, but cannot simply be deleted while
  retaining a claim to sample the same joint posterior.
- **Selection normalization dominates:** reported conditional MH shear MCSE
  is7.66e-4/1.48e-3 versus measured selection MCSE2.75e-5/1.99e-5. Declared
  selection ladders pass. MCSE can underestimate error when exploration fails;
  it does not measure shared missing-mode bias.

Chain-mean target-prior-plus-kernel spreads and late drift are heterogeneous;
some chains have large changes even when the population median is modest.
These scalar values exclude variable-dimensional environment prior factors;
they must not be interpreted as comparable full-scene joint posterior heights
or as estimates of posterior mass across different environments.

The known numerical-generator versus analytic-prior interpolation difference
remains a model-closure caveat. It has not been established as the cause of
between-chain disagreement or ruled out as a contributor to recovery error.

## Wider shear reuse fails independently

The broad profiles reuse every16th retained state from both replicas, giving
12288 saved weight draws per observation. Raw weight ESS below is
(sum w)^2/sum(w^2), without an autocorrelation correction; actual effective
sampling information can be lower. Pareto-tail diagnostics likewise cannot
certify missing modes. No Pareto smoothing was applied to the likelihood.

| Trial g1 | <20 objects with raw ESS<400 | <22 objects with raw ESS<400 |
| --- | ---: | ---: |
| .018 and .022, each | 0/128 | 0/128 |
| .015 | 8/128 | 1/128 |
| .010 | 78/128 | 26/128 |
| .040 | 89/128 | 72/128 |

At .010, median raw ESS is219 below20 and3037 below22;59/128 and21/128 have
Pareto k>.7. At .040, median ESS is58/254, and88/128 and79/128 have k>.7.
These diagnostics identify inadequate overlap in the observed samples; failure
to explore the reference posterior can make the unseen problem worse. See
[importance-sampling diagnostics](https://arxiv.org/abs/1507.02646).

The exploratory fits .01410259 and .01077937, against injection .02, therefore
are not accepted recovery or calibrated bias estimates. Broad support must be
established with multiple reference-shear posteriors, overlapping bridges and
independent checks. A denser evaluation grid using the same unsupported weights
cannot repair missing support.

Even the .002 finite-difference stencil is not an infinitesimal derivative:
on80 final states, comparing its score to the .00025 stencil changes the score
by median .62 and90th percentile19.71. Step-halving differences decrease but
remain heterogeneous. Negative finite-stencil information from unconverged
chains is not proof of intrinsic model nonconcavity. Refine the actual profile
and check stencil stability before relying on local curvature.

## Reporting fix and validation

`scripts/galsbi_bright_math.py::profile_fit` previously used a three-node
quadratic peak but linear crossings on the raw grid. Some intervals could be
missing or exclude their own interpolated peak. It now uses the same curve for
both: the quadratic on the peak segment and linear interpolation outside,
with connected threshold crossings found from the peak using Brent's method.
Peak estimates and local curvature are unchanged. The intervals remain coarse
approximations until the underlying likelihood and grid are validated.

A regression test uses a narrow off-grid quadratic peak and checks its exact
68% interval and peak containment. Five focused tests pass. Scheduled full
suite16794283:383 passed,1 skipped in101.46s. Ruff passes for the four diagnostic
scripts, helper and focused tests. GPU diagnostics completed successfully.
Original frozen reports and source snapshot were not edited. Derived corrected
intervals are saved separately in `profiles/corrected_intervals.json`.

## Recommended next bounded experiment

Follow-up: a later deep dive found a two-mode trap. Tempering was tried
first; see [tempered MH](TEMPERED_MH.md#finding--october-2-2026).

Use a discarded pilot to learn a **frozen curved proposal**, then test it with
fresh independent chains against the original sampler on a small stress panel.
A concrete transformation is u=Sersic logit and r=y-f(u), with y=(magnitude,
log radius, distortion logit, phase in a consistent chart). Moving in u while
preserving residuals follows the fitted curve. The triangular translation has
unit Jacobian; any additional position-dependent scaling needs its determinant
in the exact transformed target. Wrapped angular proposals need consistent
periodic density handling. Alternatively a small frozen mixture proposal can
use an explicit forward/reverse proposal density in MH.

Keep an ordinary full-support local component and retain exact original
prior/likelihood evaluation. Pilot optimization selects proposals only; it must
not select posterior weights, discard observations or replace the prior.
Measure per-object shear-functional ESS per GPU second, independent-replica
agreement, prefix stability and actual movement across the curved relation.
Training a proposal on pooled nonconverged states is permissible as a pilot,
but its apparent fit is not a validation reference. Test on additional objects
before making a population performance claim. Reserve tempering for barriers
that remain after the geometry repair.

After local sampling passes, test overlapping reference-shear bridges over
the fitted interval, with consistency checks in both directions and independent
replicas; retain the cut-specific selection normalization. Only then revisit
recovery and the hybrid routing rule. Do not relax convergence gates or drop
failures to reach the timing target.

The bright32k sampler costs about22.5 GPU seconds per galaxy per replica. A
separate representative16k run costs12.07 seconds. The new path diagnostics
make an efficiency improvement plausible, but establish neither a speedup nor
convergence at10 seconds. No20k or automatic longer run was launched.

## Reproduction and receipts

Source bright root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_bright_mh_20261002`.
Audit root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_convergence_audit_20261002`.
Audit `source_snapshot` links the original immutable bright snapshot. Executed
diagnostic copies live in `scripts/`; checksums are in `audit_sources.json`.
The workspace chain-audit copy subsequently received formatting-only changes;
the executed copy is retained separately. All heavy analysis ran under Slurm;
one GPU at a time was used by this audit.

| Job | Purpose | Elapsed | Result |
| --- | --- | --- | --- |
| 16794252 | Saved-chain CPU audit | 1m49s | completed0:0 |
| 16794276 | Exact replay, geometry, proposal and shear probes | 1m54s | completed0:0 |
| 16794287 | Weight audit, figures and corrected intervals | 52s | completed0:0 |
| 16794283 | Full CPU tests | 1m58s | completed0:0 |
| 16794310 | Conditional curved-path optimization | 1m14s | completed0:0 |

Key receipts under the audit root:

- `chains/chain_audit.json`, `panel.json`, `panel_draws.npz`.
- `geometry/replay.json`, `proposal_geometry.json`, `proposal_probes.json`,
  `crossed_paths.json`, `shear_stencils.json`, `result.json`.
- `profiles/overlap.json`, `corrected_intervals.json`, `chain_traces.png`,
  `shear_profiles.png`, `curved_paths.png`.
- `ridges/result.json`, `jobs.json`, `logs/`.

The three figures visualize the saved audit outputs. The curved-path plot is
only129 points per object and was rendered locally after the scheduled
optimization; all target-density evaluation ran on the allocated GPU.

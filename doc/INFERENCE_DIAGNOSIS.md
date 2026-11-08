# Inference diagnosis: context, evidence and next discriminating tests

**Scope correction:** this document reviewed the older `dev` checkout, not
the latest branch requested by the owner. Its conclusions apply only to the
historical evidence below. The corrected current review is
[the October 18 galsbi-tempering diagnosis](../.claude/worktrees/galsbi-tempering/doc/INFERENCE_DIAGNOSIS.md#scope-and-conclusion--october-18-2026),
which includes the million-pair experiment and subsequent sampling fixes.

## Review scope and conclusion — October 18, 2026

This is a retrospective review of the September 25–October 2 inference work.
The October 3 work-log entries concern the README figure. It combines the
recent log with experiment documents and small saved reports; it does not
report new sampling, a current scheduler check, or a new calibration result.
The [receipt audit](#saved-receipts-and-validation) records which numerical
results were checked directly during this review.

**We have localized several numerical failures, but not a single cause of
the shear-recovery discrepancy.** Finite-atom integration has demonstrable
errors. Continuous MH still fails to estimate shear-sensitive expectations
reliably. Its present wide-shear reuse also has poor observed weight overlap.
Curved morphology/shape geometry is a supported mechanism in inspected cases,
not an established explanation for every failed object or the recovered-shear
shift. Population and generator/target consistency remain separate questions.

## 1. Context of the problem

### The scientific quantity and its numerical requirements

For a scene s, observation y_i and shear g, the intended selected likelihood is

```text
K_i(s,g) = P(U | s,g) f_disk(y_i | s,g,U)
I_i(g)   = integral p(ds) K_i(s,g)
Z_W(g)   = integral p(ds) P(U | s,g) P(y in W | s,g,U)
log L(g) = sum_i log I_i(g) - N log Z_W(g).
```

The retained model is V3.9-like F8/C8/original physical-moment disk response
with Möbius composition. Usability and final measured selection are distinct.
The base cuts are measured radius >0.6 arcsec and magnitude <25.8; the recent
bright tests replace the magnitude threshold by 20 or 22. No truth-property
cut or removal of numerically difficult observations is part of these tests.
See the [target definition](CONTINUOUS_SAMPLING.md#exact-statistical-target),
[model identity](V39_MODEL.md#decision-and-identity) and
[selection conventions](CONVENTIONS.md#3-population-and-selection-order).

Two numerical approaches have been investigated. The finite catalogue method
approximates weighted sums over allowed scenes. Continuous MH estimates
posterior expectations at a reference shear g0, then uses

```text
I_i(g)/I_i(g0) = E_{posterior at g0}[K_i(s,g)/K_i(s,g0)].
```

This identity needs both adequate posterior exploration and usable overlap
with the trial shear. Selection normalization is still required separately.
The current finite FS2 catalogue and continuous GalSBI F24 scene law are
different priors. Moving between them changes the statistical model as well
as the numerical method; it is not a controlled test of discretization alone.
See [the hybrid prior boundary](HYBRID_SAMPLING.md#why-directly-joining-the-current-priors-is-a-different-model).

There are therefore three successive questions:

1. Can we evaluate the likelihood of a specified population accurately?
2. Do mock generation, inferred prior and measured-selection normalization
   describe that same population and observation process?
3. Does inference under that model recover shear across independent samples,
   and does the model describe the image-simulation population?

The latest work primarily diagnoses the first question. Success there would
enable the other tests, not answer them automatically. The retained model's
development response comparison is a different estimand from an inferred
shear discrepancy. Its favorable aggregate result did not establish independent
acceptance or robust 0.3% calibration; see
[development evidence](V39_MODEL.md#development-evidence) and
[the definition of m](CONVENTIONS.md#7-multiplicative-bias).

### Why the investigation moved toward continuous sampling

Earlier finite-prior runs showed strong draw-budget dependence, a small bright
tail with large influence, and indefinite local information. Increasing the
observation count did not fix these integration problems. Continuous sampling
was introduced to explore scenes between fixed atoms, but its own posterior
exploration then became the immediate obstacle. A proposed hybrid still needs
both methods validated under one common prior.

The initial performance target is 10 GPU seconds per galaxy per replica,
including initialization and output. It is an engineering target, not an
accuracy criterion. The paused 20k continuous-population study has not become
an accepted next step merely because smaller jobs completed.

## 2. Evidence so far

### Finite-catalogue integration and local inference

| Experiment | Observed result | What it establishes and what it leaves open |
| --- | --- | --- |
| V3.9 same-prior 500k model mock | The reported one-step relative g1 error moved from -9.035% at 8192 draws to +4.457% at 16384. | The draw ladder is not converged. More galaxies do not repair per-object numerical error. Neither endpoint is an accepted bias measurement. |
| Actual ConstGold 20k first pass | Information was indefinite at every draw rung. The 135 measured-mag<20 rows supplied 93.8% of net negative curvature along the final negative eigenvector. | The observed instability is concentrated in a small bright tail. This attribution does not distinguish numerical integration, finite-prior resolution, model mismatch or legitimate local nonconcavity. The requested BFGS updates never began. |
| Matched-input model mocks and bright-prior expansion | Matched mocks reproduced a curvature failure; expansion could make aggregate information positive while producing very large, unstable Newton proposals. | Image-model mismatch is not necessary to reproduce every failure. These mocks start from already selected inputs and apply selection again, so they are not ordinary parent-population closure. Expansion changed both prior support/weights and numerical behavior. |
| Exhaustive atom checks | For influential mock row 4460 under the expanded prior, sampled g1 information was -233,919 versus exhaustive +165,074. Other exhaustive focal rows retained negative information. | Numerical integration error is directly established, including a sign error. It does not account for all observed local negative curvature. |
| Fresh-parent 5130 model mock, after bright integration repair | All 36 arm/seed/rung information matrices were positive. The remaining 8k-to-16k g1 changes were 0.000165–0.000421, predominantly from measured-mag>=20 rows. | Off-catalogue generation does not invariably produce negative information. Bright-only repair does not certify the rest of the population; these were fixed-centre steps, not converged estimates. |
| Scene-disjoint prior A/B profiles on 128 fresh-parent rows | After exhaustive replacement of one four-row group, coarse preferred shear nodes still differed: A=(-0.00626,0.01735), B=(0.02374,-0.00265). | Prior-realization sensitivity remains after this repair. Only four rows were exhaustive, the grid spacing was 0.01, and the remaining numerical error is not bounded by seed agreement. This is suggestive evidence for finite-bank sensitivity, not its isolation. |

Sources: [500k result](V39_MODEL.md#completed500k-model-mock-result),
[actual first solve](V39_ACTUAL20K_BFGS.md#status-on-september-27-stopped-at-the-first-solve),
[matched-input limits](V39_MATCHED_INPUT_MOCK.md#interpretation-limits),
[expanded-step forensic audit](V39_POPULATION_AUDIT.md#expanded-step-forensic-audit-on-september30),
[paired exhaustive probes](V39_POPULATION_AUDIT.md#completed-paired-exact-curvature-probes),
[fresh-parent result](V39_POPULATION_AUDIT.md#completed-fresh-parent-first-pass-diagnostic),
[exact A/B group profile](V39_POPULATION_AUDIT.md#completed-exact-group-profile).

Two controls prevent overly simple interpretations. The original prior closely
matched audited parent marginal distributions, while the added bright catalogue
introduced some out-of-domain mass; matching marginals does not establish joint
population equality. Also, the apparently reassuring A-generated A/B control
retained intrinsic counterparts of all three bright generating inputs in B,
with large posterior mass. It was not an independent intrinsic-galaxy holdout.
See [population audit](WORKLOG.md#2026-09-27--deep-audit-separates-bright-prior-domain-cohort-conditioning-and-integration)
and [A/B support audit](WORKLOG.md#2026-09-29--independent-profiles-complete-target-the-remaining-numerical-failure).

The optimizer also contributed a distinct failure. The first historical-method
recovery used plain Newton updates although later successful historical work
used BFGS and different retrieval/carry procedures. On two saved passes, local
curvature predicted too little score change and produced an oversized Newton
correction. Repairing that procedural mismatch would not repair unconverged
scores. See [the recovery correction](V39_RECENTER_20K.md#historical-procedure-correction-and-saved-moment-audit).

### Continuous-prior sampling: what has already been tried

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Prior-proposal independence MH with exact likelihood compression | Bright-panel acceptance was 0–0.098%; all four magnitude strata failed exploration. | Compression preserved the likelihood but did not make broad-prior proposals effective. |
| Persistent environments, SED jumps and joint morphology moves | Some previously stuck directions improved; the later 262k-step four-object experiment still failed full convergence. | Specific proposals can help without providing an accepted sampler. Aggregate acceptance hides slow conditional regions. |
| Pooled covariance and joint target/environment proposals, matched 65k budgets | Both variants failed; improvements varied by object. Some joint environment moves accepted only 0.1–3%, and one chain accepted 0/1966 attempts. | A larger Gaussian or a broad joint move has already been tested and is not a demonstrated general repair. |
| Measurement-informed initialization | Better than prior-bank starts at 16k, but all four objects still failed at 64k in the completed two-replica report. | Better starts help; initialization alone has not solved exploration. This compared initializer packages, including ranking, not centring alone. |
| Original MH extended to 1,048,576 steps, four independent replicas | Combined checks passed for objects 0, 2 and 3. Object 1 still had worst Rhat=1.04247 and minimum bulk ESS=298.50; its two local shear-ratio Rhats were 1.01513/1.01799. | Longer runs help some objects. The unresolved object fails on shear-sensitive quantities too. The overall experiment remains unaccepted. |
| Complementary-ensemble stretch pilot | One difficult object's ratio Rhat improved, another worsened; functional ESS remained inadequate. | No uniform improvement or accepted fallback. Interacting walkers must not be counted as independent chains. |

Sources: [independence MH](CONTINUOUS_SAMPLING.md#summary-independence-mh-results--september-30-2026),
[persistent scenes](CONTINUOUS_SAMPLING.md#persistent-scene-mh-pilot--october-1-2026),
[262k outcome](WORKLOG.md#2026-10-01--overnight-comparison-completed-convergence-still-fails),
[pooled/joint outcome](WORKLOG.md#2026-10-01--pooled-and-joint-environment-pilots-completed-neither-accepted),
[initializer design](CONTINUOUS_SAMPLING.md#measurement-informed-initialization-comparison--october-1-2026),
[million-step design](CONTINUOUS_SAMPLING.md#longer-original-mh-chains--october-1-2026),
[stretch result](HYBRID_SAMPLING.md#current-mh-comparison-relevant-to-the-fallback).
Completed 64k and million-step results above were read directly from the
[saved reports](#saved-receipts-and-validation); older design sections still
contain historical pending-status text.

These small four-object pilots and the later bright population test have
different observation IDs and protocols. Their outcomes cannot be combined
as independent repetitions of one experiment.

### The most relevant population test: October 2 bright recovery

The test used 128 observations with measured magnitude <20 and 128 with <22,
with 18 shared rows: 238 distinct observations, not 256 independent objects
or disjoint magnitude bins. Two independent replicas each ran four chains
for 32768 steps, discarding 8192 warmup. Injection was (g1,g2)=(0.02,0);
recovery varied g1 only, with g2 fixed at zero.

| Quantity at 32k steps | Magnitude <20 | Magnitude <22 |
| --- | ---: | ---: |
| Full convergence checks passing | 0/128 | 0/128 |
| Both local shear-ratio checks passing, at .018/.022 | 35/128 | 10/128 |
| Exploratory g1 fit | 0.01410259 | 0.01077937 |
| Conditional MH contribution to shear MCSE | 0.000766 | 0.001478 |
| Selection contribution to shear MCSE | 0.0000275 | 0.0000199 |
| Raw weight ESS<400 at g1=.010 | 78/128 | 26/128 |
| Raw weight ESS<400 at g1=.040 | 89/128 | 72/128 |

All local .018/.022 observed weight ESS values exceeded 400, yet most objects
failed the local chain-agreement checks. This makes the failure relevant to
the likelihood, rather than merely a slow nuisance coordinate. Conversely,
the broader reuse develops weight concentration even within the explored
states. Both facts invalidate the current recovery calculation.

The fits are not measurements of multiplicative bias. Their uncertainties
are conditional on exploration assumptions already failing, the original
interval interpolation was inconsistent, and the wider profiles lack reliable
overlap. The two cohorts share data. Neither their apparent displacement from
0.02 nor agreement between related outputs can quantify model bias here.
See [completed bright test](BRIGHT_MH_TEST.md#completed-result--october-2-2026)
and [wider-shear audit](MH_CONVERGENCE_AUDIT.md#wider-shear-reuse-fails-independently).

Selection ladders passed their stated screens and the measured selection MCSE
was much smaller than the MH MCSE. This deprioritizes *sampled normalization
noise* as the leading observed error in this test. It does not prove the
population law or normalization formula matches mock generation exactly.

Runtime also missed the initial target: about 22.5 GPU seconds per distinct
galaxy per replica for bright 32k sampling, about 24 seconds including wider
profiles/selection. A separate representative 1024-object 16k workflow cost
12.07 seconds. These are different protocols, not a controlled speed comparison;
neither timing establishes convergence at that cost.

### What the deep audit actually localizes

Exact reconstruction of 80 saved endpoints reproduced prior and likelihood
values with zero difference; summary discrepancy was 2.84e-14. The inspected
acceptance, Jacobian and state-copying paths revealed no defect. This reduces
the evidence for those particular bugs; it is not proof that every transition
or the full scientific target is correct.

Morphology acceptance near 30% coexisted with small proposal steps relative
to explored variation. A post-hoc ten-object panel and four conditional path
probes found curved geometry. For object 80, a straight path dropped 52.27
log-density units below its lower endpoint, while an optimized curved path
between the same endpoints had no sampled drop. Redshift, SED and environment
were held fixed. See [path experiment](MH_CONVERGENCE_AUDIT.md#direct-conditional-path-experiment)
and [proposal audit](MH_CONVERGENCE_AUDIT.md#simply-widening-a-gaussian-is-not-a-general-repair).

This demonstrates a route that a straight proposal can miss. It does not
measure how wide that route is, its integrated probability, its connectivity
when nuisance variables vary, or the masses of possible other modes. A
profiled high-density curve can be too narrow to carry much posterior mass.
The selected examples cannot establish how prevalent this mechanism is.

Earlier faint-object traces also showed conditional environment trapping:
full-refresh acceptance could collapse in faint-target/bright-neighbour
regions despite healthy aggregate acceptance. These cases remain relevant
to any population-wide fallback. See
[conditional environment audit](WORKLOG.md#2026-10-01--conditional-environment-trapping-and-a-joint-move-pilot).

The interval interpolation defect was repaired and regression-tested; neither
sampling failure was repaired by that reporting change. Finite-stencil scores
also changed materially with shear spacing in endpoint probes. Smaller spacing
reduces truncation error but can amplify density/integration noise, so shrinking
the stencil alone is not an accuracy guarantee.

## 3. Reflection: have we localized the problem?

### Established failures versus remaining explanations

| Assessment | Strength of the current evidence |
| --- | --- |
| Finite-atom numerical integration is inaccurate in important cases. | Direct exhaustive comparisons, including curvature sign changes. |
| Current continuous chains do not reliably estimate some nearby-shear expectations. | Independent-chain disagreement on the actual likelihood-ratio functions, including after very long runs on one pilot object. |
| Current broad shear reuse is unreliable. | Concentrated observed weights and failed profile diagnostics. The amount that would remain after fixing reference-posterior exploration is unknown. |
| Curved geometry contributes to difficult exploration. | Supported by traces, proposal scales and conditional routes in inspected cases; no controlled successful repair has yet isolated its effect. |
| Finite-prior resolution matters beyond retrieval error. | Plausible from concentrated atom mass and residual A/B sensitivity, but remaining integration errors and overlapping source galaxies prevent a clean isolation. |
| Population mismatch, target/generator mismatch, or response-model error causes the shear shift. | Unresolved. The numerically failed recovery cannot identify their contribution. |

The two continuous-sampling failures are distinct *requirements* to fix, not
yet experimentally separated causal explanations for the final shift.
Poor exploration can itself distort observed overlap diagnostics. Even a
converged reference posterior may need additional reference shears; that must
be tested after local sampling is reliable.

Local negative observed information is also not a diagnosis of a wrong model.
A mixture likelihood can be nonconcave at a particular centre. Positive
expected information at the true parameter does not require every realized
sample or trial centre to have positive observed curvature. The finite-prior
probes establish some negative finite-stencil curvature, not the location or
correctness of a global maximum. A Newton step cannot substitute for tracing
an accurately evaluated likelihood profile.

Finally, the implemented continuous target needs precise wording. Persistent
environment refresh is defined relative to the numerical environment
generator, while target-galaxy MH uses an analytic native density. The
generator's interpolated inverse CDF is an acknowledged approximation to that
analytic law. This is not automatically an incorrect environment MH ratio:
it can define a coherent target with the numerical environment law. The open
question is whether observation generation, target density and Z_W describe
the same joint law to the required accuracy. Small marginal CDF errors alone
do not settle that question for narrow likelihoods. See
[implemented target](CONTINUOUS_SAMPLING.md#persistent-scene-mh-pilot--october-1-2026)
and [sampling-density caveat](CONTINUOUS_SAMPLING.md#a-simple-count-changing-move-partial-poisson-refresh).

### Where to look next, and what each test would decide

**First, make the target-consistency check explicit.** Record the actual
target-galaxy law, environment law, forward generator and cut-specific
normalizer in one specification. Use a bounded refinement or
sampling-versus-density check in likelihood-relevant regions, and retain
small soluble transition tests. This can proceed independently of proposal
development. Its purpose is to prevent improved mixing from being mistaken
for closure under a subtly different population. It should not change the
frozen production prior silently.

**Then test curved proposals as a controlled hypothesis.** Fit a proposal on
a discarded development pilot, freeze it, retain exact MH correction and a
full-support component, and compare against the current sampler at fixed
budgets with fresh independent replicas. Keep the prior, likelihood,
observations and selection fixed. Include the difficult geometry cases,
environment-tail cases and a predeclared held-out sample across brightness.
Do not fit and validate solely on the four optimized paths.

The primary readouts should be shear-ratio Rhat/ESS, agreement between
independent replicas, prefix stability, and functional ESS per GPU second.
Keep the existing Rhat<1.01 and bulk ESS>400 screens; they are necessary
diagnostics, not proof of missing-mode absence or sufficient population shear
accuracy. Require numerical uncertainty small relative to the relevant shear
uncertainty as in the bright protocol. Movement along a curve and higher
acceptance are supporting diagnostics, not the acceptance outcome.

If this improves local shear expectations on held-out objects, geometry has
stronger causal support as a practical bottleneck. If conditional slices
improve while full-scene chains do not, target/environment or nuisance coupling
becomes a more specific next suspect. If separated regions persist, test
mode-connecting proposals or tempering; do not assume an enlarged covariance
or another unconditional chain-length increase will resolve them. A soluble
conditional problem can validate a proposal, but is not a full-scene posterior
reference.

**After local sampling passes, test shear transport separately.** Sample at
multiple reference shears with overlapping ranges. Check ratios in both
directions, consistency through intermediate references and independent
replicas. Refine the actual likelihood profile and test stencil/grid stability.
Preserve the matching measured-selection normalization. A denser grid of the
same unsupported weights cannot establish overlap.

**Only then return to recovery and population questions.** Use independent
mock populations, selection applied once, and both shear components before
calibration claims. Repeat finite-bank size/realization comparisons with
adequate integration if discretization is the question. If evaluating a hybrid,
generate its atom bank from the same population as its continuous target and
audit faint objects too. Keep numerical failures in the sample; routing them
to more reliable integration is different from removing them.

The next useful milestone is a small panel with reproducible, numerically
stable shear likelihoods under a precisely specified target. Another large
catalogue or a favorable fitted shear would not supply the missing evidence.
These are proposed discriminating tests; this review launches none of them
and does not reopen the closed model-training investigations.

## Saved receipts and validation

The compact [review receipt](evidence/inference_review_20261018.json) stores
SHA256 identities and extracted values from these original reports under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/`:

- `galsbi_bright_mh_20261002/report.json`: final cohort fits, convergence,
  normalization checks and conditional MCSEs.
- `galsbi_convergence_audit_20261002/chains/chain_audit.json`: local-ratio
  pass counts and acceptance summaries.
- `galsbi_convergence_audit_20261002/geometry/replay.json`: exact endpoint replay.
- `galsbi_convergence_audit_20261002/ridges/result.json`: conditional path drops.
- `galsbi_convergence_audit_20261002/profiles/overlap.json`: observed weight ESS
  failures at the specified shear nodes.
- `galsbi_long_mh_20261001/report/report.json`: completed million-step results.
- `galsbi_measurement_init_64k_20261001/report/report.json`: completed 64k results.

This review checked saved report consistency, not raw-chain convergence anew.
Earlier finite-catalogue numbers above are cited from their preserved audits;
large atom sums, neural-model evaluations and source-array hash scans were
not repeated. Validation comprises small JSON checks, local links/anchors and
patch whitespace. No inference code, defaults, models or jobs changed, so no
numerical test suite was rerun. Existing unrelated workspace changes remain.

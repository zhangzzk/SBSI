# V3.6 inference review and 100k launch gate

## Decision — 2026-09-21

**Not ready for production. Do not submit the 100k inference yet.** The owner
authorized two GPUs total and conditioned launch on honest, robust checks.
This review covers the dedicated disk path, not the legacy additive default.
No model, likelihood, population cut, prior weight, shear offset, or covariance
regularization was changed during this review.

Exact-tail array16617850 and CPU reducer16617867 completed. Its worst32
objects were selected from the failed500k curvature audit. Improving agreement
on those objects cannot establish independent numerical acceptance or model
calibration. The subsequent
[likelihood-focused diagnostics](#likelihood-focused-diagnostics--2026-09-21)
demonstrate strong finite-prior and finite-step sensitivity in bright failures.

The completed [matched-truth corners](#matched-truth-conditional-corner-plots--2026-09-21)
for four selected MAG_AUTO 17–21 problem cases show no extreme individual
marginal discrepancies at their actual truth and shear. This is encouraging
for those conditional predictions, but does not validate joint calibration,
shear derivatives or population inference. It does not change the launch gate.

## Findings by layer

| Layer | Evidence and disposition |
| --- | --- |
| Artifact identity and cache provenance | Dedicated driver checks all three pinned artifacts, input/subset/config hashes, feature order, observing conditions and implementation identity. The only cache/runtime exception is the explicitly audited, replayed two-entry LRU change. Retain these guards. |
| Selected-density algebra | Correct conditional-on-selection structure: sum prior × usability × transported joint density, divided by the full-prior selected mass. No extra per-atom selected-mass division. Unit/Jacobian tests support the implementation; this is not an empirical calibration result. |
| Disk transport | Inverse Möbius map and inverse Jacobian are included. Velocity depends on unchanged radius/flux and truth pairs, not measured shape. The joint Jacobian is triangular. Exact tree-cell pooling and the two-entry response cache are numerical reuse, not fitted response rescaling. |
| Shear state | Intrinsic shape is transformed at fixed circularized radius. Classifier features and usability are recomputed per shear node. The zero context is a storage optimization, not a zero-shear likelihood. |
| Observations | Plus leg selected independently; no truth cuts; finite physical-disk shapes, positive radius/flux, strict measured radius>3pixels and MAG_AUTO<25.8. Unmatched catalogue rows are reported by preparation. |
| Model population | **Unresolved support gap.** Retained joint evidence is for the rendered secondary-role true-r<26 parent, whereas this inference is uncut. The completed identity join finds54231/500000 (10.8462%) frozen observations at true-r>=26, with zero unmatched rows. This differs from the10.89% eligible-population fraction. No independent model acceptance result exists. |
| Prior construction |24m uniformly sampled without replacement from139,936,000 rows; equal weights1/24m, no old truth-cut weights. All configured nearest20/10arcsec directed edges for a sampled primary survive, whether or not its neighbour is sampled. This is the configured finite pair model, not all physical neighbours without a cap. |
| Prior representativeness | **Open.** Disjoint case IDs and uniform sampling are necessary but do not prove generator equivalence or finite-prior convergence. Source prior cases20000–20199 contain699680 rows/case, versus approximately699568 in historical image cases. The documented historical-runtime discrepancy must be assessed, not corrected by matching row counts. |
| Selection normalization | All24m atoms contribute;64 randomized-QMC flow draws/atom estimate the measured-cut probability with CRN across shear. The object named `ExactPopulationNormalization` is exact in atom coverage, **not an exact flow integral**. No64→128→256 or independent-shift convergence check currently establishes derivative accuracy. |
| Sampling | Exact stratum plus importance-weighted complement is algebraically sound. **Original production proposal fails a real-model exact-sum check**: over half the centre mass of one bright object was in atoms with negligible proposal probability. Nested draws, two seeds and high typical ESS did not detect this. |
| Local derivatives | **Open.** h=.001 with neural FP32 evaluation has no completed step-size/precision acceptance test. The17-input classifier still contains five hard annulus features alongside four smooth features; shear-dependent boundary crossings can create nonsmooth atom probabilities. “Smooth-crowding” does not imply a globally differentiable likelihood. |
| Inference solve | **Failed at the existing centre.** Final16k aggregate information has eigenvalues[-3.6871e6,7.1478e7]. Exact integration of two problem objects still gives indefinite individual curvature. Individual non-concavity is allowed; aggregate positivity at an arbitrary starting point is not guaranteed. A validated local maximum/score solve is needed, not clipping eigenvalues. |
| Uncertainty | Current row sandwich is conditional on fixed models, finite prior, normalization and expansion procedure. It does not include training or integration error and does not automatically account for shared-scene dependence. Case-cluster/bootstrap assessment is required. |

## Likelihood and sampling audit

For fixed auxiliary outputs a=(radius,flux), define

```text
t_j(a,g) = radial_tanh(b_j(a) g)
f_j(y|g,U) = f_flow,j(T_inverse(y;t_j), context_j(g)) |J_inverse|
B(g) = sum_j pi_j pU_j(g) Pr_flow,j(a passes | context_j(g), U)
p(y|selected,g) = 1_selected(y) sum_j pi_j pU_j(g) f_j(y|g,U) / B(g).
```

Shape transport leaves a unchanged, so omitting that transport from the
radius/flux-only selection integral is correct. A future shape cut would
invalidate that simplification; the adapter rejects it. Existing density
methods follow the bundle's standardized-output density convention, omitting
a fixed output-unit determinant. It cancels from shear scores/Hessians within
this pinned model; do not call these absolute physical-unit log densities or
compare differently standardized models without restoring that determinant.

For deterministic exact support C and draws J_m from the full proposal q,

```text
L_hat = sum_{j in C} pi_j pU_j f_j
      + (1/M) sum_m 1[J_m not in C] pi_Jm pU_Jm f_Jm / q(J_m).
```

The code coalesces duplicate draws and retains prefix multiplicities. Draws
inside C contribute zero to the complement; q is not silently renormalized
outside C. The defensive prior component guarantees support, not practical
coverage of narrow likelihood peaks. Unbiasedness of L_hat does not imply
unbiasedness of log(L_hat), scores or Hessians at finite M.

Reviewed sources: `catalogue_disk_likelihood.py`, `disk_response_transport.py`,
`catalogue_disk_response.py`, `catalogue_disk_cache.py`, `crowding.py`,
`disk_inference_store.py`, `selection_normalization.py`, relevant
`catalogue_sampling.py`/`catalogue_null.py` paths and
`scripts/run_disk_inference.py`. Regression tests cover inverse/Jacobian
identities, disk density normalization, auxiliary marginal preservation,
state-major feature construction, exact sums and stratified reductions.
These local tests do not substitute for population-level validation.

## Empirical-tuning boundary

- Keep all three artifacts, cuts and prior masses fixed. No offsets, response
  multipliers, truth-based exclusions, bright-tail deletion, covariance
  inflation chosen to hide a discrepancy, or injected-shear-based recentering.
- Proposal dispersion floors (global1st percentile with a scale-based fallback)
  and robust coordinate scaling are **proposal-only heuristics**. They do not
  modify the target, but their finite-budget errors must still be checked.
- Gaussian, Student(df3), Cauchy(df1), and K variants tried on the worst objects
  are exploratory. Freeze a candidate before evaluating reserved numerical
  holdouts. If a holdout informs another change, retire it as validation and
  reserve another panel; report failed variants, not only the winner.
- More atoms/draws, an exact fallback, or a safeguarded optimizer can be valid
  numerical changes if tested against the same objective. Positive curvature
  or an estimate close to injected shear is not a criterion for choosing them.
- No automated handoff from the development-tail reducer to100k production.
  Passing unit tests, process exit codes, or a partial tail replacement is
  insufficient for launch.

## Predeclared acceptance checks

These are prospective requirements, not claims that current results pass.
The numerical shear budget is **min(6e-6,0.1×case-based SE)** per component:
6e-6 allocates10% of the historical0.3%-at-|g|=.02 calibration scale to
numerics. It is an error budget, not a target answer or a fitted correction.
Assess errors at an established stable solution; the current indefinite
matrix cannot define a trustworthy induced-shear metric. Report sampling
uncertainty in validation errors; an underpowered panel is inconclusive.

1. **Domain/prior:** audit actual frozen object membership and prior support;
   establish what evidence supports the uncut population, including r>=26 and
   role/generator differences. Existing development bias estimates do not
   establish this. Do not change the population to pass this gate. Compare
   nested12m/24m and disjoint atom banks with consistent normalization; check
   a larger bank where necessary. Exact24m sums alone cannot pass this gate.
2. **Numerator integration:** freeze a proposal using development only; compare
   its16k output against all24m atoms on a reserved uniform panel and fixed
   magnitude-stratified stress panel, at two fresh proposal seeds. Use known
   sampling weights for any population extrapolation; never treat the stress
   panel as representative. Require agreement of log-likelihood differences,
   scores and information, not ESS alone. Validate the residual contribution
   to the inferred shear within the numerical budget with adequate precision.
3. **Normalizer and derivatives:** compare64/128/256 draws and independent
   random shifts, using CRN across shear within each comparison. Compare
   h=.0005,.001,.002, compiled/eager evaluation and higher-precision reference
   density evaluation on representative and bright-tail objects. Include
   classifier bin-boundary sensitivity. Propagate numerator and denominator
   changes jointly into the same shear budget.
4. **Solve:** use a safeguarded objective/score-based search if one step is
   insufficient. Recompute genuine classifier probabilities and normalization
   at new nodes; never extrapolate the frozen cache silently. Check convergence
   from independent data-defined starts, a negligible remaining step, and a
   positive local information matrix at the accepted maximum. Investigate
   competing maxima and nonquadratic profiles. No ridge/pseudoinverse patch.
5. **Uncertainty and closure:** assess scene-level dependence using case-cluster
   influence/bootstrap and stability to influential scenes without deleting
   them from the estimator. Verify the implementation on likelihood-generated
   data at both signs/axes; distinguish this self-consistency test from actual
   image-model adequacy. Report numerical, prior and model limitations apart
   from statistical error. All preceding numerical error sources together,
   not each in isolation, must fit the stated budget.

## Reserved rows and execution

`scripts/audit_disk_population.py` verifies source hashes and joins the frozen
observations to their original truth identities on a CPU allocation. It audits
prior r>=26 membership and the corresponding usability-weighted mass at cached
shear nodes. That mass is **not** the measured-selected or posterior mass.

It reserves64 numerical-validation rows using seed20260922:32 uniform plus
8 in each measured-magnitude bin(<20,20–22,22–24,24–25.8), excluding the first32
and all50 previously inspected worst rows. These are held out from proposal
development only: they belong to the already-evaluated500k catalogue and are
not fresh independent model validation. No likelihood values are evaluated by
this reservation step.

It also reserves100000 uniformly drawn original rows with seed20260923,
**without excluding any tail or development objects**, for two persistent
one-GPU workers of50000 rows each after acceptance. This defines100k total,
not100k per GPU. Reservation is not job submission. The new sample requires
explicit row-ID support and a correctly recorded data-defined centre; do not
silently relabel the old500k-centred cache as a new100k preparation.

Population audit/full-suite job16617937 requests two CPUs,12GiB and no GPU.
Resource detection is allocation-bounded: whole-node CPU/memory recommendations
do not override Slurm limits. Existing16617850 consumes the two allowed GPUs.
No third GPU or production dependency is submitted.

### Completed CPU audit

Job16617937 completed in2m14s. Full suite:297 passed,2 skipped in41.31s.
The report is under the existing inference run's
`population_review_16617937/report.json`; source/output hashes are recorded.
All100 observed cases match their frozen source truth hashes and identities.
The24m prior has16387409 (68.2809%) atoms at true-r>=26, with no prior/observed
case overlap. These atoms contribute about47.14% of cached usability-weighted
prior mass before measured selection. That is not their selected mass or
posterior mass, and is not itself proof of bias. It makes extrapolation a
material evidence gap rather than a negligible population detail.

The reserved100k sample includes10944 true-r>=26 objects; no cuts were added.
Its sorted original-row array SHA-256 is
`4fa47d9a6d35f5f6cd08f904cfe30238af3c0273cae69e9085d6d1204018bdf8`.
The64 numerical holdouts remain unevaluated. Neither reservation closes any
scientific acceptance gate.

## Likelihood-focused diagnostics — 2026-09-21

Owner requested deeper examination of a possible likelihood failure.
`scripts/probe_disk_likelihood_failure.py` ran two persistent GPU workers
(array16621068,11m21s/10m39s), scanning all24m atoms for two already inspected
bright failures and original rows0/1. These are **four development examples,
not a representative or independent validation sample**. No production
model, cut, prior weight or cached implementation changed. Numerical holdouts
remain unused.

### Exact finite-prior sensitivity

Effective atoms means `1/sum(posterior_atom_weight**2)`, from all24m terms,
not importance-sampling ESS. The two halves are disjoint global-index-parity
partitions, each12m atoms, renormalized to unit prior mass. Their results below
are **numerator scores**, not independently selection-normalized estimates.

| Observed row | MAG_AUTO | Effective atoms at center | g1 score, even12m | g1 score, odd12m |
| --- | ---: | ---: | ---: | ---: |
|142230|17.3015|6.29|19.16|1314.33|
|3563|18.4495|16.58|-5.96|1534.90|
|0|24.3244|11636.98|1.831|1.567|
|1|25.1582|292088.13|1.289|1.334|

At the +g1 node a single atom carries84.59%/93.13% for the two bright rows;
at one corner the shares reach97.44%/96.88%. Across the existing, selected
worst32 panel, every +g1 node has a majority atom (range50.86–98.62%,
median88.88%); all32 dominant atom identities differ. This is not a single
bad-atom explanation. The leading atoms examined here are genuinely bright
(true magnitudes17.28/18.34), not the separate true-r>=26 domain gap.

These results establish serious sensitivity to the finite atom bank in these
examples. They do not distinguish physically appropriate narrow conditional
density from an overly sharp/misplaced learned density, nor establish full
population convergence. More importance draws cannot remove this dependence
on the underlying atom bank.

### Flow-context localization and derivative reference

For each observation the fixed union of the top32 atoms at each original node
was evaluated at smaller steps. The union captures93.69%/71.69% of center
mass for the bright rows, versus3.27%/0.57% for rows0/1. **These truncated
stencils are not full-prior results; omitted mass and finer-node capture are
not controlled.** Component-freezing experiments retain the same value at
the center but are diagnostic counterfactuals, never replacement models.

Recomputing the genuine classifier at every tested point, then comparing with
its probability held at the center, barely changes the million-scale bright
curvature. Holding disk transport at the center likewise leaves the effect.
Holding the flow's sheared input context fixed removes that large effect.
This localizes the dominant mechanism to the flow-context/atom-mixture part
in these examples; it does not clear classifier/transport accuracy elsewhere.

`scripts/probe_disk_derivative_reference.py` (job16621104,36s) uses genuine
FP64 input shear, preprocessing and neural evaluation with the same stored
weights, holding classifier probabilities fixed. The production helper uses
FP32 raw-shape shear arithmetic even when neural weights alone are promoted.
Independent autograd derivatives of the truncated numerator give:

| Row | g1 score at h=.001 | Autodiff g1 score | Autodiff I11 | I11 with flow context frozen |
| --- | ---: | ---: | ---: | ---: |
|142230|957.16|390.91|-1,520,954|2.60|
|3563|1520.06|766.12|-3,231,728|-5.64|

At h=.00003125 the corresponding g1 scores are391.73/767.84: convergence
toward the derivative is visible, but no production error budget is passed.
The large negative curvature therefore survives FP64/autodiff on the fixed
union; it is not merely FP32 cancellation. The original h=.001 derivative is
also demonstrably inaccurate for that same function. No full-population
autodiff result or new-node selection normalization was computed.

The finite-step component decomposition at h=.000125 supports rapid changes
in atom responsibility as the dominant g1 mechanism: mean component I11 is
positive (~61545/~117368), but component-score variance is much larger
(~1.582m/~3.349m). Their difference is negative. Finite-step identity residuals
are nonzero and reported; these figures are not exact autodiff decompositions.
Some individual components also have negative curvature in another direction.

The physical-moment response artifact explicitly documents a first-order
moment objective using finite h=.2 labels. Its successful mean-response
checks, and retained improvement in selected NLL, do not validate the full
shear-dependent density. Existing percent-level variance discrepancies remain
relevant. No evidence here justifies declaring the learned flow physically
correct or incorrect, or modifying it to make an inferred shear agree.

Reports live under the frozen run's `likelihood_probe_16621068/`: per-row exact
and truncated results, input-hash-tracked reducer `summary.json`, and
`derivative_reference_16621104.json`. The reducer includes finite-difference
mixture decomposition and its identity residual; that residual must not be
silently interpreted as an exact derivative identity or used as a correction.

CPU report job16621160 completed in43s after **302 tests passed,2 skipped**
(27.37s). Original report job16621083 was cancelled after stalling before
Python started on a different compute node; its logs are preserved. No
production inference or independent model acceptance was launched.

## Joint measured-property density plots — 2026-09-21

The owner requested joint likelihood plots with the problematic observations
overlaid. `scripts/plot_disk_joint_likelihood.py` evaluates all six pairwise
**slices** through the four-output density (e1,e2,radius,flux), holding the two
unplotted outputs at their observed values. These are not pairwise marginals.
The summation uses the fixed33-atom union from each prior diagnostic, with
original uniform prior and shear-dependent usability weights, and divides by
the existing full-prior selection normalizer. Posterior weights are not reused
as mixture weights. Output-conditioned response coefficients are recomputed as
radius/flux vary; the inverse transport Jacobian and fixed physical-unit
determinant are included. Input shear, preprocessing and neural evaluation
use FP64; classifier values remain the original cached FP32 values.

The center is g=(.01391227336,.00028266795); the overlay uses g+(.001,0).
Contours mark density ratios .001,.01,.1,.5 relative to each center-shear
slice's peak, not credible regions. Pink crosses and lines mark the measured
values. At the observation, the two truncated contributions change by
delta-log-density1.82430 and2.87779 (~6.20x and17.77x). The respective
center/shifted union captures from the previous all-atom diagnostic are
93.69%/98.94% and71.69%/97.82%. Those captures are production-precision
reference values, not independently recomputed full-FP6424m sums.

Broad97x97 grids completed in job16621509; local97x97 grids completed in
16621540. Local half-widths are .015 in e1,
.01 in e2, .1arcsec in radius and15% of measured flux. These change only the
display, not the likelihood, population or model. Raw grids, inputs and
provenance are retained next to PNG/PDF figures under `doc/figures/`.

Final figures: [row142230](figures/v36_joint_likelihood_20260921_final/row_142230_joint_slices.png),
[row3563](figures/v36_joint_likelihood_20260921_final/row_3563_joint_slices.png).
The `overview/` subdirectory contains broad views; PDF versions are adjacent.
Final rendering provenance links back to the untouched source grids in
`v36_joint_likelihood_16621540/` and `v36_joint_likelihood_16621509/`.

**Scope:** these are truncated-prior contributions to the selected likelihood,
not full24m surfaces. Off-observation atom coverage is unknown. Neither these
visualizations nor the prior-selected examples establish model adequacy.

## Fainter problematic objects: marginal corner plots — 2026-09-21

The requested follow-up uses rows412689,120835,86325 (MAG_AUTO19.4465,
20.1248,20.5881). They were already in the exact worst32 panel, selected by
production negative-curvature contribution rather than by image appearance.
For each, freeze the union of top32 atoms across the existing nine shear
nodes. Generate8192 forward draws per atom at the current center and
center+(.001,0), with common random numbers keyed by atom identity. Apply the
actual output-conditioned disk response, followed by the unchanged measured
cuts. Each draw receives its atom's original prior-times-usability weight;
normalize over selected draws within this restricted atom set.

These are true sample-based marginals over unplotted outputs, not the previous
fixed-output slices. Diagonal histograms show1D densities; lower panels show
2D histograms and approximate68%/95% highest-density contours, without KDE or
smoothing. Histogram thresholds are computed against total selected sample
weight, not renormalized to the plotting window. Limits cover weighted0.1–99.9
percentiles plus the observation and a small margin. Plot coordinates are
measured e1,e2,radius in arcsec,and MAG_AUTO.

**Interpretation boundary:** this is the normalized predictive distribution
conditional on an observation-chosen atom subset, not the full24m law and
not an independently valid goodness-of-fit test. Neither posterior atom
weights nor any new magnitude19 cut are used. The reported top32 share of
likelihood at the observed point does not measure integrated predictive
coverage. No fitted model or production inference setting is changed.

Job16622565 used oneA40/fourCPUs/64GiB. Five focused tests passed; generation
failed on a non-finite or non-positive radius/flux forward draw before response
evaluation. No corner figures were completed and no invalid draw was repaired.
The invalid fraction and originating atom remain unmeasured. The owner then
asked whether the plots condition on each observation's actual truth: they
do not. This mixture follow-up was paused and superseded by the completed
[matched-truth diagnostic](#matched-truth-conditional-corner-plots--2026-09-21)
below. The preceding mixture plots should not be interpreted as that diagnostic.

## Matched-truth conditional corner plots — 2026-09-21

The owner clarified the diagnostic: condition on the actual galaxy's intrinsic
simulation properties and its own rendered neighbour scene, and overlay its
actual measured outputs. This supersedes the paused prior-mixture corner task.
Four cases were frozen before evaluating the model, selecting the closest
MAG_AUTO to17.5/18.5/19.5/20.5 in each unit bin among the existing worst50
approximate-production negative-curvature contributors. This does not make
them representative or establish that each is an exact-likelihood failure.

| Frozen row | Case / input ID | MAG_AUTO | Observed marginal CDFs: e1, e2, radius, MAG_AUTO |
| --- | --- | --- | --- |
|239507|99 / 301899|17.6253|88.18%,84.70%,35.49%,47.95%|
|440648|51 / 495396|18.4570|34.50%,40.28%,45.60%,35.58%|
|274111|55 / 61948|19.4945|58.03%,22.46%,44.26%,46.81%|
|10642|67 / 119656|20.5649|52.55%,42.83%,29.04%,65.60%|

`scripts/plot_disk_truth_corners.py` checks source truth/crossmatch/shapes
hashes against the frozen image-mock manifest, rechecks all four detection
matches and measurements, and compares intrinsic morphology with original
generated catalogues. No unmatched selected row is silently discarded.
Own truth supplies ellipticity, Sersic index, true magnitude and intrinsic
circularized radius. Simulation position angles are explicitly converted from
degrees before the generic coordinate helper. All rendered neighbours within
7arcsec supply the original three crowding summaries; the response uses up to
20 nearest nonself neighbours strictly within10arcsec. No truth cuts are made.
All four sources have actual shear(.02,0), applied once to intrinsic shape.
The V3.6-like pinned common flow is evaluated in FP64 and composed with the
retained output-conditioned disk response. At fixed truth, the classifier
is a scalar and cancels from the density conditional on usability.

Each plot uses262144 forward samples, no prior-atom integration, and no
measured MAG_AUTO/radius analysis cuts. Diagonal histograms and lower-triangle
2D marginals use64 bins, with approximate68/95% density contours; no KDE or
smoothing. Windows cover0.1–99.9% marginal quantiles plus the observed point.
All original contexts, response-neighbour IDs/distances, source hashes,
random seeds, marginal CDFs, contour window masses and draw archives are saved.
One draw for the17.6253 case failed the numerical open-disk check before
transport (1/262144, about0.00038%); it was recorded and excluded without
clipping. Other samples were valid; no post-transport failure occurred.
Thus that panel is normalized over numerically valid samples, explicitly
annotated. The unchanged inference cuts would accept99.9950%,99.9928%,100%,
100% of the valid samples respectively; those cuts were not applied here.

OneA40 job16622674 completed in1m16s including five passing focused tests.
Source samples/report: `doc/figures/v36_truth_corners_16622674/`.
Presentation-only CPU job16622696 completed in29s using hash-verified saved
samples. Final products:

- [MAG_AUTO 17.6253](figures/v36_truth_corners_20260921_final/row_239507_truth_corner.png)
- [MAG_AUTO 18.4570](figures/v36_truth_corners_20260921_final/row_440648_truth_corner.png)
- [MAG_AUTO 19.4945](figures/v36_truth_corners_20260921_final/row_274111_truth_corner.png)
- [MAG_AUTO 20.5649](figures/v36_truth_corners_20260921_final/row_10642_truth_corner.png)
- [Machine-readable report](figures/v36_truth_corners_20260921_final/report.json)

PDF counterparts are alongside the PNGs. The report links the original saved
draws; presentation-only re-rendering did not change their values.

**Reading:** these actual-truth conditional distributions look substantially
more regular than the previous atom-mixture slices, and the actual measurements
are not extreme in any plotted 1D marginal. Some asymmetric size/flux tails
remain. The change of conditioning and marginalization matters: the old
figures cannot be used as direct evidence for misshapen own-truth conditional
flow predictions. These four development examples do not validate the joint
likelihood, shear derivatives, population normalization or prior integration,
and do not determine whether sparse bright-object training is a problem.
No model, cut or production inference setting was tuned.

## Limitations and next decision

The review has found concrete blockers; it has not established a production
acceptance result. The completed worst-tail and likelihood probes inform the
next numerical/model checks, not close the domain/prior/normalization/solver
gates. Prior integration and derivatives must be converged before a failed
shear recovery can be attributed to physical model mismatch. Separately, the
flow's conditional distribution and shear response need image-level validation
in the bright regime; a same-model closure test alone cannot establish that.
If the uncut population needs new model training or a different scientific
population, return to the owner for that decision. Do not restart archived
joint-training or independent-validation launchers as an inferred remedy.

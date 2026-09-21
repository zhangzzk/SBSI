# Work log

## 2026-09-21 — Atom map: where the mass is versus where the proposal looks

Owner asked for a picture of the prior atoms of one observation, on their
predicted measurements centred on the observation and on their true
properties, with drawn atoms marked separately from undrawn ones and coloured
by posterior mass. Added `scripts/plot_proposal_atom_map.py`,
`jobs/job_plot_proposal_atom_map.sh` and `tests/test_proposal_atom_map.py`.

Diagnostic only. No flow was evaluated and no GPU was requested: the figure
reuses the cached proposal coordinate table, the cached zero-shear detection
probabilities and the exact 24m centre-node posterior weights already on disk,
so it does not consume the owner's two-GPU limit. The target, model, cuts and
prior are untouched.

The draw is reproduced from `DefensiveLocalProposal.select_priority_batch`:
key `q_j / u_j`, generator seeded by `_priority_row_seed(8701, row)`, keep the
16384 largest. The tilted-stratified exact stratum is not excluded, so the
drawn set is the proposal's own race rather than the estimator's final
bookkeeping.

Results, centre node, worst-curvature rows:

| row | measured mag | atoms drawn | captured mass reached | at defensive floor | mass at floor | mass rank of the proposal's top atom |
|---|---|---|---|---|---|---|
| 142230 | 17.30 | 9/32 | 15.1% | 9 | 32.6% | 3 |
| 409188 | 18.35 | 23/32 | 53.8% | 3 | 32.5% | 3 |
| 3563 | 18.45 | 24/32 | 89.2% | 3 | 2.6% | 2 |

Findings.

- The failure is a hard priority threshold, not wasted duplicate draws. For
  row 142230 the race cuts cleanly: every atom with `q >= 7.6e-6` was drawn
  and every atom with `q <= 4.9e-6` was not. No floor atom was drawn, which is
  what 16384 draws from 24m predicts.
- Row 142230's two heaviest atoms, 51.0% of the posterior between them, both
  sit at the defensive floor `4.1667e-09`, tied with the 24m atoms the
  proposal treats as impossible. Its single favourite atom, `q = 0.742`, is
  only the third heaviest and carries 8.6%.
- The owner's blending argument is visible in the figure. On row 142230 the
  mass-carrying atoms scatter over roughly +-0.3 in predicted `g1`/`g2`
  relative to the observation, while the same atoms sit in a tight clump near
  zero in predicted size and flux. Intrinsic shape shows no concentration at
  all, and drawn versus missed atoms are interleaved in it.
- Atoms that matter are bright and large (mag 16.5-18.5, `Re` 0.9-2.9 arcsec)
  and sit in a sparsely populated corner of a prior whose bulk is mag 24-28 at
  `Re < 0.5`.
- Counting atoms found is misleading. Row 409188 drew 23 of 32 yet reached
  only 53.8% of the mass, because the atoms it missed were the heaviest and
  the largest in `Re`.
- Row 3563, a magnitude fainter and less blended, behaves well on the same
  machinery: mass-carrying atoms cluster around the observation in predicted
  shape and 89.2% of the mass is reached.

Correction to the 2026-09-21 proposal-coordinate entry. The production draw is
priority sampling without replacement, so a dominant `q` takes one slot, not a
proportional share of the 16384 draws. `S = sum_j w_j^2 / q_j` as reported
there is the with-replacement second moment; the priority estimator's variance
runs on inclusion probabilities `min(1, q_j / tau)`. The variant ordering is
expected to be unaffected, since tempering helps by lifting starved atoms over
`tau`, but the absolute ESS figures in that entry describe a sampler that is
not the one in production.

Revision, same day, at the owner's request. Three changes to the figure and
one new measurement.

- The observation's own true properties are now plotted, as a dashed crosshair
  in both truth panels. They are recoverable after all: the case input feather
  named in `input/image_mock_manifest.json` `per_case[].sources.truth.path`,
  indexed by `source_input_index` and checked against its `index_input`
  column. Intrinsic ellipticity and circularized radius are rebuilt from axis
  ratio and position angle with the atom convention, `|e| = (1-q)/(1+q)` and
  `Re * sqrt(q)`, and agree with the stored atom columns to 1e-8. Values:
  142230 `e = (-0.2660, -0.0657)`, `Re_c = 1.446"`, `r = 17.275`, case 114;
  409188 `e = (-0.1508, +0.1036)`, `Re_c = 1.586"`, `r = 18.329`, case 58;
  3563 `e = (-0.0214, +0.1610)`, `Re_c = 1.168"`, `r = 18.335`, case 113.
  In the size/brightness truth panel the crosshair sits among the
  mass-carrying atoms, so those atoms are not off in an implausible corner.
- Drawn atoms are now split by why they were drawn. An atom whose mixture
  probability is at the defensive floor `delta/24m` won the race on its
  uniform share alone; anything above the floor was ranked by the proxy. They
  are drawn in two colours at their true counts.
- Every undrawn atom, background or mass-carrying, is now the same small dot,
  so size no longer codes for anything but "drawn". Legibility is carried by
  colour, opacity and a hairline edge instead.

New measurement, the draw-class split:

| row | drawn, ranked by the proxy | drawn on the uniform floor | atoms above the floor in 24m |
|---|---:|---:|---:|
| 142230 | 3,980 | 12,404 | 1,231,271 |
| 409188 | 1,828 | 14,556 | 1,435,576 |
| 3563 | 2,454 | 13,930 | 1,540,103 |

Most of the draw is the defensive lottery, not the proposal: 76%, 89% and 85%
of the 16384 slots go to atoms the proxy scored at the floor. The proxy is
effectively spending only a few thousand slots on its own ranking, which is
consistent with the concentration defect reported above and means the 0.1
defensive fraction is doing more of the work than its name suggests. This is a
count, not a variance statement; it does not by itself say the floor draws are
wasted.

Validation. `tests/test_proposal_atom_map.py`, 33 passed inside the job. Job
16625903 COMPLETED in 00:01:34, partition `cluster`, no GPU; supersedes
16625836 and 16623681, which produced the same numbers with the earlier
figure. Two earlier attempts failed and were repaired: 16623593 on the
detection cache being point-major `(10, 24000000)` rather than atom-major, and
16623655 on `exact/row` being present in only 2 of the 32 panel records. Both
failure modes are now covered by unit tests. Figures and machine-readable
reports copied to the owner's checkout at `SBSI/plots/proposal_atom_map_row*.png` and
`SBSI/plots/proposal_atom_map_row*_report.json` (that directory is not tracked in git).

Limitations. Mass is known only for the 32 atoms the exact calculation
retained per node, capturing 93.5%, 71.2% and 71.1% of the posterior for the
three rows; the remainder is spread somewhere across the grey background,
which is a uniform 60k subsample and is not proven massless. The three rows
are the top of the worst-curvature list and are not representative. Only the
centre stencil node is shown. The background is a uniform subsample, so a
drawn atom in a sparse region may be a subsample artifact rather than a
neighbour of the observation.

Next steps. Redo the nine-variant comparison on priority-sampling inclusion
probabilities rather than `S`, so the quoted efficiency matches the production
estimator. Quantify, rather than eyeball, the shape-versus-size split by
comparing standardized proposal distances in the two coordinate pairs for
drawn and missed atoms.

## 2026-09-21 — Proposal-coordinate comparison on the worst-32 exact panel

Owner asked whether the tilted proposal should rank atoms on magnitude and
radius alone instead of all four measured coordinates, reasoning that
anisotropic blending shifts the e1/e2 centre of the likelihood. Added
`scripts/compare_proposal_coordinates.py`,
`jobs/job_compare_proposal_coordinates.sh` and
`tests/test_proposal_coordinate_compare.py`.

Proposal-only diagnostic. `q` is the only object varied; the target, model,
cuts, prior weights and production settings are untouched, and every
contribution still carries the exact `pi_j / q_j` correction, so each variant
is an equally valid estimator of the same quantity. No flow was evaluated and
no GPU was requested: the run reuses the cached proposal coordinate table
(`disk_assembled_v1/proposal/coordinates.npz`) and the exact 24m centre-node
posterior weights in `tail_exact_16617850/`, so it does not consume the
owner's two-GPU limit.

The script drives the production `WholeCatalogueProxy` through a shim that
supplies exactly the five attributes it reads, and inverts the defensive blend
to recover the bare softmax so components can be mixed. Variants are convex
mixtures of local components, each with its own coordinate subset, dispersion
inflation and softmax temperature. Reproduction of the recorded production
proposal is asserted, not assumed: `prod_4d` matches the 32 shared atoms of
`proposal_probe_16617724/retrieval.json` to a maximum relative difference of
2.40e-13.

Metric is the importance-sampling second moment `S = sum_j w_j^2 / q_j` over
the 32 exact top atoms per row, a lower bound on the full-prior second moment
because the retained atoms capture 0.68--0.99 of the centre-node posterior.
Lower is better. Geometric mean over the panel, paired against production:

| variant | geom-mean S | median S | worst S | rows better |
| --- | ---: | ---: | ---: | ---: |
| prod_4d (production) | 5588 | 8336 | 3.50e7 | -- |
| rf_2d (radius+flux) | 9127 | 1.36e4 | 3.49e7 | 14/32 |
| prod_4d_T2 | 2449 | 1223 | 1.90e7 | 24/32 |
| prod_4d_T3 | 4707 | 2895 | 1.08e7 | 20/32 |
| prod_4d_T4 | 9295 | 4480 | 9.66e6 | 17/32 |
| rf_2d_T2 | 6438 | 4015 | 1.45e7 | 19/32 |
| mix_4d_rf2d | 5168 | 7513 | 3.49e7 | 16/32 |
| mix_4d_rf2d_T2 | 2725 | 1668 | 1.61e7 | 25/32 |
| shape_sigma_x3 | 6193 | 8551 | 3.46e7 | 15/32 |

Findings. First, the owner's mechanism is confirmed but is not where the gain
is. Dropping the shape coordinates helps exactly where production fails and
hurts where it works: the correlation between `log S_prod` and
`log(S_rf2d/S_prod)` is -0.541 (p=0.0014, n=32); the nine brightest rows
(MAG_AUTO<18.5) improve by a geometric-mean factor 0.88 while the other 23
degrade by 2.08. On its own `rf_2d` is worse on average.

Second, the dominant defect is proposal concentration, not coordinate choice.
Production puts a median 0.54 of its mass on a single atom, and for row 142230
one atom takes 0.742 while 12 of 33 important atoms sit at the defensive floor
`delta/24m`. Tempering the existing softmax addresses this directly: T=2 gives
a paired ratio 0.44 (68% interval 0.33--0.58, p=0.007) and raises the median
effective-sample-size bound at 16384 draws from 2.0 to 13.4.

Third, adding the two-dimensional component on top of tempering is not
established. `mix_4d_rf2d_T2` improves the most rows (25/32) and is best on
the bright subset, but against `prod_4d_T2` alone the paired ratio is 1.11
(p=0.181) over the panel and 0.87 (p=0.423) on the nine bright rows. With
n=9 that comparison is underpowered and neither direction is demonstrated.

Limitations. The worst-32 panel was selected by production negative-curvature
contribution, so it is neither random nor held out and the absolute S values
are not population numbers; only the ordering of variants on identical rows is
supported. S is a lower bound restricted to the retained atoms, and the
effective-sample-size figures are correspondingly upper bounds. The
comparison is at the existing centre only and says nothing about scores,
information, derivative accuracy or the separate domain/prior and
normalization gates. No temperature was adopted and no production setting was
changed.

Validation: 16 focused tests passed in 18.49s, including reproduction of an
explicit Gaussian mixture through the production proxy class, inversion of the
defensive blend, and convex-weight checks. Job 16623076 completed in 3m23s on
`cluster` (8 CPU, peak 7.5 GiB); superseded runs 16623034 (failed on an
attribute name, now covered by a test) and 16623045 are preserved. Ruff,
compilation, Bash syntax and `git diff --check` passed. Report:
`proposal_coordinate_compare_16623076/report.json`.

Next steps: bracket the temperature between 1.5 and 3 against scores and
information rather than the centre-node second moment alone; repeat on a
reserved uniform panel before any production change, since the present panel
cannot support an acceptance decision.

Note on repository state: the entire V3.6 disk pipeline was uncommitted in the
working tree when this work started (`dev` at 63b8102 predates
`catalogue_disk_likelihood.py`, `disk_inference_store.py`,
`output_conditioned_response.py` and `crowding.py`). Those files were verified
byte-identical to the implementation hashes recorded in
`row_142230_exact.json` and committed on this branch so the pipeline exists in
git; they are unchanged from the owner's working tree.


## 2026-09-21 — Consolidate completed plot diagnostics and current stopping point

Updated this worklog and `doc/V36_INFERENCE_REVIEW.md` after delivering the four
matched-truth corner plots. Removed stale pending-output/clarification wording
from the failed prior-mixture attempt, preserving its failure and provenance.
Added the completed matched-truth result to the review's opening decision and
linked the final figures and machine-readable report from its results section.

Current conclusion: four selected problem cases at MAG_AUTO 17.6253, 18.4570,
19.4945 and 20.5649 show no extreme individual marginal discrepancy when
conditioned on their own simulation truth/neighbours at actual shear (0.02, 0).
This is not a representative validation sample, a four-dimensional joint
calibration test, or a test of shear derivatives. The older prior-mixture slices
answer a different question and do not directly diagnose the own-truth flow.
No inference cuts, model artifacts, prior weights or production settings changed.
The 100k production launch remains withheld; no new computation was submitted
for this documentation update. Next scientific work remains the unresolved
prior/proposal, derivative, normalization and model-coverage checks described in
the [review's next decision](V36_INFERENCE_REVIEW.md#limitations-and-next-decision).

Validation: read the final JSON report and checked all four row identities,
draw counts and marginal CDFs against the recorded results; all eight final
PNG/PDF files exist. `git diff --check` passed. Documentation only; no tests
rerun (the generation job's five passing tests remain recorded below).

## 2026-09-21 — Own-truth conditional corners across MAG_AUTO17–21

Owner clarified that the requested plot conditions on each galaxy's actual
simulation truth, not a diagnostic prior-atom mixture. Added
`scripts/plot_disk_truth_corners.py`, `jobs/job_plot_disk_truth_corners.sh`, and
`tests/test_disk_truth_corners.py`. Frozen rows239507/440648/274111/10642 have
MAG_AUTO17.6253/18.4570/19.4945/20.5649: nearest each one-magnitude-bin midpoint
among the existing worst50 approximate-production curvature contributors,
chosen before evaluating predictions. They are illustrative development cases.

The script verifies source hashes, frozen row identities, truth/detection joins,
actual measurements, and intrinsic morphology against generated catalogues.
It builds each primary's eight flow features from its rendered full scene,
explicitly converts degree angles, applies actual shear(.02,0) once to intrinsic
shape, samples the retained flow in FP64, and composes the output-conditioned
disk response using the nearest20 nonself neighbours within10arcsec. Crowding
uses all rendered neighbours within7arcsec, without truth cuts. At fixed truth
the usable-event classifier is a scalar and cancels from the law conditional
on usability; no prior mixture or measured analysis cuts are used here.

262144 draws per object yield 1D marginal histograms and 2D histogram-based
68/95% contours. No KDE, model tuning or clipping. Invalid numerical draws are
counted explicitly before/after response and excluded from the displayed valid
sample law; their frequency is retained on each figure and in JSON. Samples,
contexts, pair IDs/distances, source hashes and marginal CDFs are saved.
Matplotlib/resource-check skills informed layout and the oneA40/fourCPU/64GiB
scheduled allocation; user queue was empty before submission. Ruff, compilation,
Bash syntax and `git diff --check` passed. Job16622674 completed in1m16s,
with five focused tests passing in22.02s; model loading plus generation/plotting
took31.8s. All source/identity/morphology/measurement checks passed, with no
selected-object match dropped. Source draws/report live at
`doc/figures/v36_truth_corners_16622674/`. One of262144 draws for row239507
failed the pre-transport open-disk check; all other draws were valid, and no
post-transport failures occurred. No radius/flux invalidity occurred here.
The four observed marginal-CDF ranges are35.5–88.2%,34.5–45.6%,22.5–58.0%,
and29.0–65.6%. The plots do not show conspicuous marginal outliers at actual
truth, but do not establish joint calibration, population coverage or shear
derivative accuracy. See [matched-truth results](V36_INFERENCE_REVIEW.md#matched-truth-conditional-corner-plots--2026-09-21).

Visual inspection found legend/x-label crowding. Added a hash-verified
`--render-from` mode and adjusted only figure spacing; CPU job16622696 completed
in29s, re-rendering the exact saved samples into
`doc/figures/v36_truth_corners_20260921_final/`. Final visual inspection passed.
No production code or model was changed; the earlier mixture job stays paused.

## 2026-09-21 — Marginal corner plots for fainter problematic observations

Owner requested triangular contours with diagonal marginal histograms for
fainter failures. Added `scripts/plot_disk_faint_corners.py`, the corresponding
one-GPU launcher, and `tests/test_disk_faint_corners.py`. Chosen previously
audited rows412689/120835/86325 have MAG_AUTO19.4465/20.1248/20.5881: the worst
production-curvature row in19–19.5 and20–20.3, plus the faintest of the exact
worst32. This is a selected diagnostic panel, not a representative holdout.

For each row, retain the union of the nine-node top32 atoms from completed
exact24m references. Draw8192 physical samples per atom at the current center
and center+(.001,0), with atom-identity-seeded common random numbers. Recompute
output-conditioned disk transport for every sample. Use original equal-prior
times cached usability weights, apply the unchanged measured radius>0.6arcsec
and MAG_AUTO<25.8 cuts, then normalize within the retained atom subset. No
posterior atom reweighting, KDE, smoothing, model tuning or new cuts. This
produces genuine finite-sample1D/2D marginals, unlike the earlier density slices,
but NOT the full24m population predictive law. The top32 likelihood mass at the
observation is reported separately and is not integrated predictive coverage.

The Matplotlib skill informed the corner layout; the resource-check skill
informed oneA40/fourCPU/64GiB allocation, job16622565, with an empty user queue
before submission. Three focused tests cover mixture-selection weighting,
contour mass thresholds and plotting units; the two previous plotting tests
also run before generation. Ruff, Python compilation, Bash syntax and
`git diff --check` passed. The intended output directory was
`doc/figures/v36_faint_corners_16622565/`; no completed plot products were produced.
Job16622565 passed all five tests in24.23s, then failed on a forward draw with
non-finite or non-positive radius/flux before response evaluation. No corner
plots were completed. Invalid draws were not clipped or repaired. Their
frequency and originating atom have not yet been measured. The owner then
clarified whether these predictions were conditional on each observation's
actual truth properties. They were not: both this attempted follow-up and
the earlier slice plots use data-chosen prior-atom mixtures. This mixture work
was paused and superseded by the completed
[own-truth diagnostic](#2026-09-21--own-truth-conditional-corners-across-mag_auto1721),
using actual simulation truth and neighbour context, not those prior atoms.
No core implementation, pinned artifact, inference cache or production job
changed. The100k launch remains withheld.

## 2026-09-21 — Plot joint-density slices for problematic observations

Owner requested joint likelihood plots with observed properties overlaid.
Added `scripts/plot_disk_joint_likelihood.py`,
`jobs/job_plot_disk_joint_likelihood.sh`, and two focused tests in
`tests/test_disk_joint_likelihood_plot.py`. The diagnostic uses previously
inspected bright rows142230/3563 and their fixed node-wise top32 atom unions.
Six pairwise slices hold the two other outputs at the observation; these are
not pairwise marginals or credible regions. Original uniform-prior/usability
weights and full-prior cached selection normalization are retained. Disk
response is recomputed for each radius/flux output cell. Neural evaluation,
input shear and preprocessing are FP64; classifier probabilities are the
cached FP32 values. Contours compare the existing center and center+(.001,0).
No posterior reweighting, KDE, population cut change, or model tuning is used.

The Matplotlib skill informed the OO multi-panel layout and shared sequential
color scale. The resource skill informed a one-A40, four-CPU,64GiB Slurm job
16621509, within the two-GPU combined limit; queue was empty before submission.
Whole-node resource recommendations do not supersede the allocated resources.
Python compilation, Bash syntax and `git diff --check` passed before launch.
Initial job16621488 stopped in the new test on a2.8e-17 grid-center rounding
difference. Explicitly set each production grid center to the observed value
and use an appropriate floating tolerance for the synthetic linspace test.
Immediate retry16621500 apparently reused stale shared-filesystem bytecode
(traceback displayed the new allclose call but executed the old exact-array
assertion). A job-local bytecode cache in16621509 avoids that reuse; both
focused tests then passed in15.93s. Job16621509 completed in3m25s. Output goes to
`doc/figures/v36_joint_likelihood_16621509/`, including raw grids and provenance.
All six panel centers replay the direct observed-point density within1e-9.
Visual inspection found very narrow e1 peaks at the broad display scale and
an overlapping footer. Added a fixed local-window view and a rendering-only
mode; job16621540 re-evaluates finer local grids and redraws the saved broad
grids without changing any density. Its two focused tests passed in13.64s;
the full job completed in3m07s. The local half-widths are .015/.01 in shape,
.1arcsec in radius and15% of observed flux; these are display settings only.
Details: [joint density plots](V36_INFERENCE_REVIEW.md#joint-measured-property-density-plots--2026-09-21).
Matplotlib's outside figure legend still overlapped the explanatory footer;
changed it to an explicit figure-coordinate position. CPU-only job16621572
redraws both sets from their saved grids into
`doc/figures/v36_joint_likelihood_20260921/`, preserving the original arrays
and provenance; completed in32s. Reserved more bottom margin after inspection
to separate the legend from axis labels, with final CPU rendering job16621622
writing `doc/figures/v36_joint_likelihood_20260921_final/`. These layout-only
jobs do not load models or recompute densities. Final job16621622 completed
in32s; visual inspection confirms separated labels/legend/footer. Tests passed2/2; Ruff, Python
compilation, Bash syntax and whitespace checks pass. The plot lesson is a
narrow e1 ridge with the observation on its steep right flank in both examples;
radius/flux slices are multi-ridged. This is descriptive, not model rejection.
The diagnostic unions capture93.69%/71.69% at the center observation; captured
mass elsewhere on the plotted surfaces is unknown. No full-prior surface or
physical adequacy claim follows. No production code/cache or100k job changed.

## 2026-09-21 — Diagnose likelihood failure separately from sampling

Owner asked to look deeply into possible likelihood failure. Added diagnostic
scripts `probe_disk_likelihood_failure.py`,
`summarize_disk_likelihood_failure.py`, and
`probe_disk_derivative_reference.py`, their two GPU launchers, and
`tests/test_disk_likelihood_failure_probe.py`. No core `sbsi/` implementation,
model, production cache, population, or estimator was changed.

Protocol: two previously inspected failures (rows142230/3563) and original
rows0/1, not the reserved numerical holdouts. Sum all24m atoms at the existing
nine shear nodes, measure posterior effective atom count, and compare four
disjoint index-mod4 banks and their two12m unions. Bank comparisons concern
the numerator only: subset-specific selection normalization is not computed,
and these partitions are not independently generated priors. On each fixed
union of node-wise top32 atoms, recompute genuine classifier features and
probabilities at h=.001,.0005,.00025,.000125; compare compiled/eager density,
FP64 neural evaluation, and freezing each of probability/context/transport
at its center value. Report original-node captured mass. These truncated
numerators and deliberately frozen-component arms are not production models
or full-prior derivative validation.

A separate autodiff reference uses FP64 neural evaluation AND input shear/
preprocessing; the production helper retains FP32 raw-shape arithmetic even
when neural weights are promoted. The reference freezes the classifier and
compares finite differences down to h=.00003125 against autograd derivatives.
The reducer separates posterior-averaged component information from the
covariance of component scores, reporting the finite-step identity residual.

GPU array16621068 completed in11m21s/10m39s, using two A40s total and four
CPUs/64GiB per worker. Resource detection confirms one visible A40 per worker;
scheduler allocation, not whole-node recommendations, bounded the work.
One-GPU derivative job16621104 then completed in36s, with all five new tests
passing. CPU summary/full-suite job16621083 stalled in uninterruptible I/O
before Python started on cip-cl-compute3 (process inspection: shell child,
zero CPU, about1MiB RSS). Cancelled only that job, preserving logs. A first
replacement submission specifying inter,cip plus cip-cl-nv01 was rejected
without creating a job; CPU-only16621160 on partition cip/node cip-cl-nv01
completed in43s as the replacement. The full suite passed **302 tests,
2 skipped in27.37s**, followed by successful reduction. No100k production job
is submitted. Ruff, Python compilation, Bash syntax and `git diff --check`
pass. Reports are in the existing run's `likelihood_probe_16621068/summary.json`
and `derivative_reference_16621104.json`; all scientific GPU jobs are complete.

All-atom results: bright rows142230/3563 have6.29/16.58 effective posterior
atoms at the center, versus11636.98/292088.13 for rows0/1. The12m parity halves
give g1 numerator scores19.16 versus1314.33 and-5.96 versus1534.90. Existing
worst32 records have median88.88% single-atom mass at+g1, with32 different
dominant identities. This establishes serious atom-bank sensitivity in the
selected failures, not population-level prior convergence or model bias.

Full-FP64/autodiff on the dominant unions, with center-frozen classifier,
retains I11=-1.521m/-3.232m. Freezing flow context reduces I11 to2.60/-5.64;
freezing disk transport leaves the large effect. At h=.001, g1 scores
957.16/1520.06 differ substantially from autodiff390.91/766.12. Thus sampling,
prior resolution and finite-step errors are separate observed problems;
negative curvature on these unions is not merely FP32 cancellation. The
unions capture93.69%/71.69% of bright center mass (only3.27%/.57% for rows0/1),
so no full-prior fine-step or classifier-normalized derivative claim follows.
No normalizer was extrapolated to new nodes. Findings and table are in
[likelihood-focused diagnostics](V36_INFERENCE_REVIEW.md#likelihood-focused-diagnostics--2026-09-21).

The finite-step mixture decomposition at h=.000125 gives positive mean
component I11 (~61545/~117368) but much larger component-score variances
(~1.582m/~3.349m), accounting for the negative mixture I11. The reported
finite-step residuals are nonzero and preserved, not corrected away. This
supports rapidly changing atom responsibility as the dominant g1 mechanism
on these unions. It does not imply every individual component is concave;
some also have negative information in the other direction.

The scientific-critical-thinking skill motivated separating model adequacy,
finite-prior resolution, derivative numerics, and sampler error. The resource
skill motivated the bounded allocation and dependency. Existing moment
calibration and improved NLL are not full-density/shear-derivative validation.
Next: establish converged prior/derivative integration before attributing
failed shear recovery to physical model mismatch; separately validate bright
conditional density/response against images. No archived model training or
cancelled independent validation chain was restarted. This investigation did
not establish whether the flow is physically too sharp or correctly sharp
but inadequately integrated by this prior.

## 2026-09-21 — Review end-to-end inference and gate the requested100k run

Owner requested a full likelihood/prior/sampling review, caution about empirical
tuning, and two-GPU100k inference only when robust. Added
[review and launch gates](V36_INFERENCE_REVIEW.md#decision--2026-09-21), covering
disk change of variables, full-prior measured selection, uniform finite-prior
construction, exact-plus-complement sampling, finite-log bias, state-dependent
classifier geometry, derivative precision, solver convergence and uncertainty.
Linked it from `doc/INFERENCE.md`. **Production launch is withheld.**

The inverse Jacobian/auxiliary-marginal composition and stratified reduction
are consistent on inspection and in existing regression tests. Important
unresolved issues remain: original Gaussian retrieval has demonstrated support
failure; exact bright-object curvature can still be negative;24m prior
resolution and source-generator/runtime equivalence are not established;
selection64-QMC/h=.001 convergence has not been established. The17-input
classifier contains five hard annulus bins in addition to its four smooth
crowding features, so global smoothness must not be assumed. The normalizer
called `ExactPopulationNormalization` is atom-complete but still uses a
sampled flow integral. The standardized-output density omits a fixed physical
unit determinant that cancels in same-model shear derivatives, not in arbitrary
absolute density comparisons. No implementation/cached-science change was made.

Added `scripts/audit_disk_population.py`,
`tests/test_disk_population_audit.py`, and CPU launcher
`jobs/job_audit_disk_population.sh`. Job16617937 completed in2m14s after the
full suite passed **297 tests,2 skipped in41.31s**. Resource receipt shows no
GPU; allocation was two CPUs/12GiB, with BLAS/OMP limited to one thread.
The script checks frozen input/source hashes, rejects ambiguous or unmatched
truth joins, scans all20 prior shards and reserves rows without evaluating
new likelihoods. Output is under the existing run's
`population_review_16617937/report.json`.

Actual frozen500k support audit:54231 objects (10.8462%) have true-r>=26;
all source matches succeed. Prior:16387409/24000000 (68.2809%) atoms have
true-r>=26 and account for approximately47.14% of cached usability-weighted
mass **before measured cuts**. This is not selected/posterior mass or proof
of bias. It documents a material gap from the retained true-r<26 development
evidence; no independent model acceptance exists for this uncut population.
Prior cases20000–20199 and observed cases40–139 are disjoint.

Reserved a64-object numerical holdout (seed20260922),32 uniform plus8 per
fixed magnitude stratum, excluding first32 and all50 inspected worst rows.
It is held out only from proposal development, not independent image/model
validation. Separately reserved a uniform100000-row production sample
(seed20260923), with **no tail/development exclusions**, for two50000-row
workers after acceptance;10944 of these rows have true-r>=26. Array hash and
membership are persisted. No production job or automatic production dependency
was submitted. Subsequent proposal choices must be frozen before inspecting
holdout likelihoods; further tuning retires those holdouts as validation.

Current GPU array16617850 remains the only GPU allocation (two A40s total).
At review handoff each worker has completed three exact/comparison objects
and is progressing. It and reducer16617867 remain development diagnostics,
not launch gates by themselves. Scientific-critical-thinking guidance motivated
the development/acceptance separation and prospective numerical budget; the
resource skill confirmed allocation limits. No models, response scales,
empirical shear corrections, cuts, prior weights or production caches changed.

Validation: Bash syntax, Python compilation, system Ruff and `git diff --check`
pass. The py31 environment lacks the Ruff module; `/usr/bin/ruff` is used
instead. Next: finish the existing development-tail benchmark; separately
resolve the uncut support/prior, normalization, derivative and solver gates.
If resolution requires retraining, a changed population or restarting archived
independent validation, obtain an explicit owner decision. Task-wakeup tools
are unavailable; the existing Slurm reducer runs without waking the assistant.

## 2026-09-21 — Start exact tail validation with two persistent workers

The second all-atom benchmark16617754 completed in5m13s. Row3563 remains
non-concave under exact integration: information
`[[-2454379.98,267716.02],[267716.02,-251436.95]]`. Thus a successful proposal
repair must not be confused with proving positive full-catalogue curvature at
the current expansion point.

Student(df3) union probe16617759 completed in4m09s; Cauchy(df1) union16617785
completed in4m28s. Both preserve the original1024 exact atoms and the original
defensive complement sampler. Cauchy ranking moves the previously missed
33%/18%-mass atoms to ranks58/343. AtK65536/M16384 the worst object's g1 score
error falls from1116.814 to41.388 versus exact; the second object's error falls
from72.157 to9.829. This is a useful improvement, **not adequate evidence of
integration convergence**. No proposal variant is promoted to production.

Refactored the unchanged all-atom reduction into `exact_observation` in
`scripts/probe_disk_curvature.py` and added a blockwise-normalization regression.
Added `scripts/validate_disk_tail_integration.py`,
`jobs/job_validate_disk_tail.sh`, and `scripts/summarize_disk_tail_validation.py`.
The new panel covers the32 most negative audited observations, reuses the two
completed exact references, and compares Cauchy-union K65536/K262144 with16k
complement draws against full24m-atom sums. This is a selected failure-tail
benchmark, not an unbiased population subsample. The reducer reports errors
and an explicitly diagnostic partial replacement of32 contributions; it never
reports a shear estimate or covariance.

Array16617850 has exactly **two persistent one-A40 workers**,16 objects each,
with model/prior allocations retained and exact/comparison checkpoints saved
after each object. CPU reducer16617867 depends on array success. Allow roughly
one hour plus queue time, based on the measured exact-scan rate. The initial
startup16617841 was cancelled at56s before panel evaluation after identifying
the generic prefilter-size guard forK262144. The wrapper now declares its
original131072-location shortlist separately from the larger whole-prior union;
fresh output roots and freshly loaded code are used for16617850. No old output
was deleted or overwritten.

Both16617850 startup gates pass6 tests (10.99s/10.89s), including independent
SciPy checks of Student(df1/df3) scores, union uniqueness, and exact summation
across atom blocks. Resource receipts show two full46GB A40 allocations with
four CPUs/64GiB host memory each. Ruff, Bash syntax and `git diff --check` pass.
No other SBSI GPU work is active. Both workers have completed their first
comparison and exceeded one million atoms on their next exact observation.
AtK262144 the two existing benchmarks' g1 score errors are6.1435/3.6575,
versus1116.8140/72.1568 in production. The corresponding first-object
information errors are `[[-9803.50,428.99],[428.99,-7548.77]]`; this is a
substantial reduction but does not establish catalogue-level numerical tolerance.
Task-wakeup/automation tools remain unavailable; the Slurm reducer will run
automatically but does not wake the assistant.

Next: use the exact panel to decide whether the integration proposal is accurate
enough for a targeted recompute, and separately determine whether a safeguarded
recentered solve is required. Models, cuts, prior masses, normalizer and original
production moments are unchanged; the final positivity requirement stays intact.

## 2026-09-21 — Locate proxy support failure and test auxiliary retrieval

Whole-proxy probe16617724 completed in4m19s wall time. Increasing the same
Gaussian-proxy exact stratum toK131072 still substantially disagrees with the
all-atom result for row142230. The centre's two leading atoms carry32.90% and
18.12% of exact numerator mass but have proxy ranks961486 and1648131. Neither
is shortlisted; each has proposal probability about4.17e-9, hence only6.83e-5
expected occurrences in16k draws. This explains why even64k nested draws and
two seeds can look stable while missing most of the centre's mass.

Added `scripts/probe_disk_auxiliary.py`, its scheduled launcher and
`tests/test_disk_auxiliary_probe.py`. The diagnostic unions the unchanged
original1024 candidates with radius/flux-only Gaussian-proxy candidates, using
total exact budgets4096/16384/65536. Complement sampling and weights are
unchanged; union membership is unique and preserves every original candidate.
This is a proposed integration remedy, not a new likelihood or population cut.
Job16617745 completed in4m06s wall time; its startup gate passed both union tests
(2 passed in9.23s). Radius/flux-only Gaussian ranking also fails to recover the
dominant atoms (ranks850217 and1651496), so that proposed remedy is rejected.
Added an optional four-output Student-t(df3) ranking to the diagnostic union,
leaving the actual flow density and complement proposal unchanged. Its score
ranking is checked against SciPy's independent Student density. Dependent
job16617759 is running; startup gate3 passed in9.96s. The likelihood's physical
scales and models are not modified by this proposal-only tail choice.

The exact probe now accepts an explicit row from the audited worst-object list.
Job16617754 is checking row3563, the second-worst production row, against every
prior atom as a separate numerical benchmark. Together with16617759 this
preserves the two-GPU cap. Both proposals and benchmarks keep all observations
and the same finite prior; no full rerun has been launched.

Scheduled regression16617733:40 passed in22.50s across the existing sampler,
disk likelihood/cache/runtime-compatibility tests and new stencil regression.
Ruff, shell syntax and whitespace checks pass. The original500k moments remain
unchanged. A finite-prior/non-concavity problem also remains possible after
fixing integration; an exact benchmark, not positivity alone, gates the remedy.

## 2026-09-21 — Exact bright-object probe confirms curvature and integration error

Probe array 16617682 completed: exact arm 5m49s wall time, sampled arm 3m46s.
CPU summary 16617708 completed in 21 seconds. Added the hash-checked reducer
`scripts/summarize_disk_curvature_probe.py`; its report is
`curvature_probe_summary_v1.json` under the existing V3.6 run root.

For the worst production observation (row142230, measured MAG_AUTO17.3015),
exact summation over all24million atoms gives score `(926.5168,-472.9388)` and
information `[[-1686130.55,500735.41],[500735.41,-879639.81]]`, versus production
score `(2043.3309,-1210.4676)` and information
`[[-2707895.46,452212.67],[452212.67,-2134515.99]]`. Thus **both** genuine
non-concavity at the initial centre and material finite-draw integration error
are present for this object. The exact numerator's leading atom changes across
the stencil; atom9262565 goes from8.57% posterior mass at the centre to84.59%
at the positive-g1 node and97.44% at the positive-g1/negative-g2 corner.

The32 worst objects retain a large negative curvature contribution at16k,
32k and64k draws under both seeds8701/8702. More draws alone do not resolve the
problem. This targeted selection is diagnostic, not a population estimator.
No objects were removed and no correction/offset was fitted to injected shear.

Added `scripts/probe_disk_proposal.py` and `jobs/job_probe_disk_proposal.sh`.
Single-GPU job16617724 checks each dominant exact atom's original shortlist
membership, full-proxy rank and expected proposal hits, then compares whole-prior
Gaussian-proxy exact strata K1024/K16384/K131072 with16k complement draws on
the same64 diagnostic objects. It changes numerical integration only; models,
population, centre and full-prior normalization remain fixed. Ruff and shell
syntax pass. Startup and results pending. No production inference code or
results were overwritten, and no remedy has yet been accepted. Updated stale
introductory text in `doc/INFERENCE.md` and the runtime boundary in
`doc/JOINT_CALIBRATION.md` to distinguish the dedicated disk driver from the
unchanged default/legacy path and the stopped model-development investigation.

## 2026-09-21 — Diagnose failed V3.6 production curvature

All 20 production partitions of array 16606146 completed, covering all 500,000
observations. Combiner 16606157 failed on non-positive information at the first
draw rung. A CPU audit also finds non-positive information at every subsequent
rung, including the final 16,384-draw result: eigenvalues
`[-3687071.14567819, 71478092.02541134]`. No valid final estimate or covariance
has been published; the positivity guard remains intact.

Added `scripts/audit_disk_curvature.py`, checking partition/input hashes,
identities, exact observation coverage and moment finiteness, then separating
normalization curvature and tabulating diagnostic contributions by measured
properties and source case. Job 16617664 completed in 24 seconds; report is
`curvature_audit_v1.json` under the existing 20260920 run root. The negative
direction is dominated by bright observations, not normalization: numerator-only
information eigenvalues are `[-3874527.63, 70520765.56]`. The ten most negative
observations contribute about -19.51 million along that direction. Exclusion
tables are diagnostics, **not alternative estimates or authorized cuts**.

Corrected the interpretation of saved weight arrays: they have draw-rung and
observation axes. The final-rung median ESS is 893.97, not the 161.41 obtained
by pooling all rungs; 64 final-rung Pareto-k diagnostics are nonfinite, while
all final score/information values are finite.

Added `scripts/probe_disk_curvature.py`, `jobs/job_probe_disk_curvature.sh`, and
`tests/test_disk_curvature_probe.py`. The probe revalidates production cache,
model, input and response hashes without changing inference implementation or
old receipts. One arm sums all 24 million atoms exactly for the worst object;
the other uses paired 16k/32k/64k draws and two seeds for the worst 32 objects
plus the first 32 comparison rows. Array 16617682 has two one-A40 tasks, the
only active SBSI GPU jobs, each with four CPUs/64 GiB host memory. Resource
detection confirms full 46 GB A40s. Test job 16617683 passed the stencil-algebra
regression (1 passed in 12.15 seconds); ruff, launcher shell syntax and
`git diff --check` pass. Both probes have loaded the hash-validated runtime
in 81 seconds and are compiling/evaluating. Probe results pending. Updated
the inference and catalogue-prior status sections; old outputs remain intact.

Next: distinguish finite-draw integration failure from genuine non-concavity
of the fixed finite-prior likelihood before choosing a correction. No model,
population, normalizer, production moments, or positivity requirement changed.

## 2026-09-20 — Audit full-prior pilot and eliminate response-cache thrashing

All20 preparation shards16605612 completed in7m41s–8m19s each;
assembly16605621 completed in2m08s. Full-prior pilot16605622 completed:
24million atoms,256observations,16k draws, compiled density and16-object
chunks. All moment and diagnostic arrays are finite, all objects used16384
draws, and the summed information eigenvalues are59224.20 and119798.25.
Peak PyTorch allocation19,025,835,520B; initial load82.7s, inference257.04s.
Warm throughput201.43s/240objects projects29.14h at500k/four GPUs, excluding
startup. This exposed an implementation bottleneck, not a calibration failure.

`DiskCatalogueLikelihood._coefficients` had one cache slot, while the shared
stratified estimator alternates exact-candidate and complement grids at every
shear node. They evicted one another, recomputing identical response sums nine
times. Replaced it with a bounded two-entry LRU: no density, coefficients,
draws, probabilities, stencil or numerical precision changed. Added an
alternating-grid regression test, including bounded eviction and equality to
fresh evaluations.

`scripts/run_disk_inference.py` now permits this one audited runtime-only
change over the existing preparation receipts. It requires exact old/new
density hashes, the original producer-driver hash, identical source text for
all four preparation functions, every other implementation hash unchanged,
and every scientific identity unchanged. Receipts and arrays are not rewritten;
both producer identity and current runtime identity are recorded in results.
Tests reject other implementation/model changes. This is not a general
"ignore provenance" flag.

Validation16606079: **290 passed,1 skipped in40.45s**. The same256-observation
full-prior pilot is replayed, and `scripts/check_disk_pilot_replay.py` requires
bitwise equality of every saved array before writing its validation receipt.
Resource checks again verified a full46GB A40; the conservative96GB host
allocation and16-object chunks are retained. Ruff, shell syntax and
`git diff --check` pass. Replay16606079 completed successfully: every array
(moments, all ladder levels, draw/unique counts and weight diagnostics) is
**bitwise identical** to pilot16605622. Total inference decreased257.04s to
92.69s. Excluding each first16-object compile/warmup batch, throughput
improves201.43s/240objects to51.66s/240objects, a3.90x speedup. Peak GPU
allocation remains19.03GB. Four-GPU projection is7.47hours of warm inference;
allow8–10hours plus queue delays, subject to production throughput variation.

Added production launcher for20disjoint25000-object partitions with a combined
four-GPU throttle and a dependent strict full-coverage combiner. The launcher
requires the replay receipt and refuses implementation changes since that
validated pilot. No production job was submitted before replay validation.

**Production array16606146 is submitted**:20x25000observations, capped at four
full A40s,96GB host memory and four CPUs per worker, writing
`production_lru_v1/part_00` through `part_19` beneath the existing20260920 run
root. Dependent strict combiner**16606157** runs after array success; it refuses
missing/overlapping coverage, incompatible identities, hash mismatches and
non-positive combined information. Scientific settings remain uncut24million
atoms, measured radius>0.6arcsec/MAG_AUTO<25.8, and the full512–16384 ladder.

Startup verified for all four production workers on full46GB A40s. Partitions
0–3 passed the production gate, completed cache loading/compilation and
reported16or more evaluated objects; the latest check had304,272,256,16rows
respectively. Warm first-worker rates span roughly3.8–4.7objects/s, consistent
with the8–10hour planning range. No early error was observed. The combiner is
pending array success. Scheduled task-wakeup tools remain unavailable, so no
automatic assistant follow-up is installed; the scheduler chain continues.

## 2026-09-20 — Integrate disk inference and start full-prior cache preparation

### Driver and validation

Added `scripts/run_disk_inference.py` with prepare/assemble/run stages and
`sbsi/disk_inference_store.py` for validated subset loading, frozen shear-state
classifier probabilities, global proposal summaries and memory-mapped sharded
disk responses. It calls the existing `run_adaptive_section5` estimator;
no alternate estimator or additive-response approximation is introduced.
The legacy driver still rejects the preparation-only disk config. Default
likelihood and numerical configurations remain unchanged.

The new path verifies model, subset, feature-order, pairing, input and code
hashes. Assembly requires every source shard, rejects partial/mismatched
receipts, applies proposal dispersion floors globally, and combines full-prior
selected masses. Cached probabilities cover zero plus the nine stencil nodes;
missing nodes fail closed. Inference writes additive per-object moments and
all six512–16384 draw levels for the existing strict partition combiner.
The requested500k input fixes a common starting center
`(0.013912273363531775, 0.0002826679500570267)` without truth information.

Selection uses64QMC draws per atom, proposal summaries128, step0.001,
K1024/prefilter131072, and unmodified float64 tilted-score arithmetic.
Preparation streams use `SeedSequence([8101, source_shard])` for selection and
`SeedSequence([8201, source_shard])` for coordinates, with selection reset at
every shear point for CRN. This explicit sharded stream is not bitwise replay
of the historical single global stream. The subset and scientific cuts are
unchanged; no truth cut or empirical correction is added.

Full-A40 validation16605587: **288 passed,1 skipped in24.97s**. First attempt
16605578 stopped on an overly strict exact-equality test: NumPy target-column
indexing changes float64 reduction order by1.11e-16. The test now allows
roundoff, with no change to computed summaries. Ruff, launcher syntax and
`git diff --check` pass. Resource checks confirm full46GB A40s; workers honor
their four-CPU/48GB allocations, not the entire node's detected resources.

The real-model end-to-end smoke pilot completed on16384 prior atoms and128
real observations in23.03s of inference, peak PyTorch allocation154,863,104B.
It prepared ten classifier/selection states, reloaded hashed caches and wrote
finite moments at every draw level. This is an integration check, not a shear
calibration result or a full-prior runtime estimate. Artifacts live at
`inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/pilot_16605587/`.

### Scheduled full-prior stages

The24million subset job16603639 completed successfully in20m08s: all20shards,
24,000,000 atoms and392,263,736 retained directed pairs. The completion
manifest and per-shard receipts were checked after the owner's notification.

New jobs:

- **16605612**, `jobs/job_prepare_disk_inference.sh`:20prior shards, at most
  four full A40 GPUs total, writing `disk_cache_v1/`.
- **16605621**, `jobs/job_assemble_disk_inference.sh`:CPU assembly after all
  preparation tasks succeed, writing `disk_assembled_v1/`.
- **16605622**, `jobs/job_disk_fullprior_pilot.sh`:256observations against
  all24million atoms,16k draws,16-object chunks and compiled density, after
  successful assembly. No500k observation inference array is submitted yet.

Startup was checked on all four workers: each passed source verification and
reached at least524,288/approximately1.2million proposal-coordinate rows;
shard0 reached786,432 rows at the handoff check. All four allocations are full
A40s, split across two compute nodes. No early failure was observed.

Check full-prior memory, finite moments, compilation behavior and timing before
launching production observation partitions. Any failed predecessor blocks
its dependent stage. Automatic task-wakeup tools are unavailable in this
session; these scheduler dependencies do not themselves monitor/report results.

## 2026-09-20 — Adopt an uncut uniform 24-million-atom prior

Owner selected24million atoms instead of the full139,936,000-row store.
`scripts/subsample_disk_prior.py` samples globally without replacement,
seed20260920, assigns equal1/24million masses, records original draw ranks and
source identities, verifies source hashes and preserves every selected
primary's response neighbours (including unsampled secondary atoms).
No truth-property selection is introduced. `DiskCatalogueModelCache` accepts
explicit original identities to keep classifier pair geometry correct after
compaction. Added subsampling/identity tests and the CPU preparation launcher;
updated the16k config description and
[prior documentation](CATALOGUE_PRIOR.md#v36-like-uncut-24-million-atom-subset).
Default models, numerical defaults and the preparation-only guard are unchanged.

The earlier2–3day estimate below is superseded: it combined the enlarged prior
with a virtual24GB A40 throughput probe. A matched full-A40 probe16603474 gives
19–23hours for tilted sampling alone on140million atoms; this is not an
end-to-end ETA. New24million-atom full-A40 probe16603624 tests16/32-object
batches. Its synthetic timings must not be reported as complete inference time.
All20 full uncut source shards have completed; source preparation is reusable.

Full-A40 probe16603624 completed: warm medians0.5468s/16objects and
0.6687s/32objects, projecting1.187h and0.726h respectively for tilted proposal
sampling at500k observations/four GPUs. Peak tensor allocations are11.72GB
and22.09GB, without resident likelihood tensors. This removes the oversized
prior bottleneck; complete runtime still requires the real-model pilot.
The resource skill verified a full46GB A40 and sufficient CPU-node memory;
preparation remains sharded/memory-mapped within its48GB Slurm allocation,
not the entire node's detected memory or CPU count.

Five targeted subsampling/cache tests pass in15.04s; Ruff, shell syntax and
`git diff --check` pass. CPU job16603639 completed the full suite:
**283 passed,2 skipped in52.08s**, no regression/snapshot failures, then began
materializing `prior_subset24m`. Its startup and resource report were checked.
First shard completed with1,200,254 sampled atoms and19,625,085 retained pairs
in53.7s including source verification; the second shard is in progress at
handoff. Top-level completion manifest is written only after all20 shards.
Production integration, normalization/proposal caches and a real-model pilot
remain next steps; no full V3.6 inference has been launched.

## 2026-09-20 — Begin uncut V3.6-like 500k/16k inference preparation

### Current status and runtime decision

The500k input is complete:18,157,963 eligible measurements in cases40–139,
no crossmatch-without-shape or matched-without-truth rows. Of the eligible
population,1,977,959 (10.89%) have true r>=26 and lie beyond the V3.6 training
parent. This percentage describes the eligible population, not an exact count
in the frozen500k subsample. No selected zero shapes were found. First full
feature shard completed with6,996,800 atoms and114,354,404 directed pairs;
feature construction took219.7s before receipt hashing. Remaining CPU shards
are array16603279,1–19 throttled to four tasks, dependent on first-shard and
test success. Startup was checked through source verification and pair progress.

Density/normalizer benchmark16603280 projects4.23h for density/response work
plus3.45h for128-draw proposal coordinates and64-draw nine-node normalization
on140million atoms, on four GPUs. Crucially, full-size synthetic proposal
benchmark16603325 measured2.5847s per two observations at139,936,000 atoms:
**44.9h for tilted sampling alone** at500k observations/four A40-24Q GPUs.
An opt-in blocked float64 cumulative sum (`WholeCatalogueProxy.cdf_block_size`)
reduces the same probe to2.1235s/two observations in16603358, about36.9h.
It changes addition order, not proposal probabilities; four boundary/ragged
tests pass in16603359. Historical sampling uses the original scan by default.
The production32-object proposal batch would exceed memory on this enlarged
prior; a memory-bounded batch is required before launch. Synthetic benchmarks
exclude resident likelihood tensors and local candidate retrieval.

Thus the revised planning estimate is **2–3 days on four A40s**, not the
earlier inference-kernel-only4.4h estimate. No multi-day GPU inference or
normalization array has been submitted. Await the owner's direction on this
resource commitment; CPU truth-feature preparation may finish independently.
There is no available automation/scheduled-wakeup tool in this session, so no
automatic follow-up has been created.

### Implemented components and remaining integration

`DiskCatalogueLikelihood` now routes exact, sampled, ragged tensor and
conditional-density paths through the inverse Möbius map and its Jacobian,
with observed radius/flux-conditioned coefficients reused across stencil
nodes. `DiskCatalogueModelCache` rebuilds all17 classifier features in chunks
for every shear state, retaining probabilities instead of huge feature frames.
Tests compare direct inverse-density calculations, state-dependent mass,
padding/invalid outputs, coefficient reuse, and the original classifier feature
constructor. Full suite16603250: **275 passed,2 skipped in43.29s**; targeted
density16603200:4 passed. `configs/inference_v1_2_16k.json` records the extended
draw ladder without changing defaults; its prior remains explicitly unbound.
The disk components are not yet wired into persisted model-cache loading,
distributed preparation, the production driver or full-scale memory management.
The preparation-only guard remains essential: do not remove it until those
paths and a real-model end-to-end pilot pass. No inference result exists yet.
Final CPU regression job16603398 reports **279 passed,2 skipped in37.26s**,
with zero regression/snapshot failures. Ruff passes for all files changed by
this preparation turn; `git diff --check` and launcher syntax checks pass.
At handoff8/20 truth-feature shards are complete (55,974,400 source rows),
four CPU shards are progressing, and no GPU job is active or pending.

### Initial preparation implementation and validation

Owner requested the previous full inference with V3.6-like, no truth-property
cuts, measured radius strictly above0.6arcsec, twice the complement draws,
and up to four GPUs. This is newly authorized inference, not a restart of the
cancelled independent image-validation campaign. The four-GPU allowance applies
to this run. Current preparation uses CPUs; the bounded benchmark used one GPU.

Added preparation-only `configs/likelihood_v3_6_like.json`, pinning the three
models and strict radius/flux predicates. `run_inference.py` explicitly refuses
this config until the catalogue density/state-major classifier adapter is
integrated, preventing an accidental additive-response run. Default configs
are unchanged. Extended `OutputCut` with explicit `NAME:>LOWER:UPPER` bounds;
old half-open bounds retain their behavior and cache identity. Extended the
ConstGold adapter with explicit no-truth-cut flags and the V3.6 usability
definition, including valid zero shapes. Reports count selected true-r>=26
objects separately: the requested uncut population extrapolates beyond the
model's true-r<26 training parent. No offset or calibration correction is added.

Added `CatalogueDiskResponse`, an exact CSR pair-sum evaluator on arbitrary
atom/output triples. It pools only identical tree split cells and handles
isolated atoms as empty sums. Added an uncut, neighbour-complete truth-feature
preparer: original zero contexts are hash/identity checked; nearest20 neighbours
within10arcsec use the pinned V3.6 development pairing convention, without
truth-property cuts. This store is not yet a normalized production prior.

Validation: response pooling tests pass; Slurm16603127 reports **270 passed,
2 skipped in48.15s** for the complete active test suite. Benchmark16603029
failed on a missing PYTHONPATH;16603043 started before its parent-key correction
and failed on the old `ids` field. Both setup issues were corrected.
Successful A40-24Q benchmark16603054 verified the pinned model and source hashes:
262144 atom/output terms,3916572 pair/tree evaluations in1.06254s; nine warm
density calls in0.97518s. Linear projection is **4.422 GPU-parallel wall hours
on four GPUs for500k objects and16384 atoms**, excluding preparation, classifier
states, normalization, retrieval, I/O, queue time, and first compilation.
This is a one-scene uniform-atom eager-kernel benchmark, not a production timing
or inference result. Receipt: `logs/v36_benchmark_16603054.json`.

Preparation job16603127 first freezes500k uniformly sampled plus-leg objects
from cases40–139, seed20260914, then builds the first uncut prior-feature shard.
Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1`.
Startup was checked through passing tests and successful catalogue processing.
Production inference has **not** been submitted. Next, after the runtime
decision above: finish production integration, build all new
proposal/normalization caches and launch at most four GPU workers. Retain
K1024, h0.001, the one-step estimator, full-prior64-draw CRN normalization and
the paired8192/16384 draw ladder. Do not reuse V3.5 probabilities or cut caches.

## 2026-09-20 — Stop investigation, retain V3.6-like, and archive research code

Owner requested cancellation, the name V3.6-like for the retained single-flow/
disk-emulator/smooth-classifier set, and repository cleanup. Cancelled the
downstream jobs first (16595003,16602287,16602111,16595004,16595000,16594999),
then generation16593294 at about11:10 CEST. Accounting records 34 completed
scenes and six cancelled generation tasks; all pending stages are cancelled.
Queue inspection confirms no remaining research jobs. External artifacts,
partial products, original protocols, and scientific receipts remain intact.
The 40-scene validation produced no bias result and is closed by owner request.

Added `configs/models_v3_6_like.json` with the original three model identities,
four artifact hashes (including emulator metadata), source-manifest hash,
population/cuts, retained endpoints, and explicit cancellation status.
`sbsi.models` and CLI expose V3.6-like alongside the historical V3.5-like preset.
The new disk loader checks model/metadata hashes; the additive loader rejects
disk models. Extracted five unchanged feature functions into `sbsi.crowding`
and added a seventeen-input constructor. Updated model/API/README/environment/
convention/inference documentation and AGENTS.md. Inference defaults and model
weights are unchanged; V3.6-like is not a drop-in additive likelihood config.

Archived 453 retired files (150 scripts, 199 jobs, 90 tests, 7 plots, 4 experimental
package modules, 3 detailed documents) and 19 before-cleanup snapshots under
`archive/research-2026-09-20/`. Every archived file has an original path, archive
path, SHA-256 and byte count in `manifest.json`. All 472 hashes verified after
the moves. This preserves pre-existing uncommitted work and all original
scientific source bytes. The full previous work log and joint investigation
are archived; current docs summarize the retained model and link to history.
See [repository organization](REPOSITORY.md#research-archive).

Validation: active Python AST syntax and four launcher shell checks pass;
Ruff passes for the modified/new Python files. AST comparison confirms the
five extracted feature functions are unchanged. CPU verification 16602609
completes in 1m20s with empty stderr: `python -m pytest tests -q` reports
**267 passed, 1 skipped in 56.39s**, zero regression/snapshot failures.
`python -m sbsi validate-model V3.6-like` verifies all four artifact hashes.
The actual flow, classifier and disk emulator load; feature-order/training
metadata checks and a small flow/classifier prediction smoke pass. Active
imports have no missing or archived dependencies; all active document links
and their section anchors resolve. Final active counts are 31 package modules,
19 scripts, 4 launchers, and 35 test/support files. The verification used no GPU.
Final scheduler inspection is empty; `git diff --check` also passes.

Limitations: existing endpoints remain development evidence; fresh multi-axis
validation was stopped. Archived launchers need deliberate restoration in an
isolated checkout and source/hash checks before reuse. No science work or
automatic monitoring is scheduled by this cleanup.

## Earlier history

- [Complete log before 2026-09-20 cleanup](../archive/research-2026-09-20/doc/WORKLOG.md)
- [Joint calibration experiments](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#goal-and-retained-population)
- [Earlier research archive](REPOSITORY.md#research-archive)

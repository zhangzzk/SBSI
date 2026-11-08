# Second-component constant-shear test

## Purpose and launch record

[Protocol](rule.md#predeclared-experiment) fixes the original40 scenes and changes
only applied shear direction to g2+/-0.02. This measures R12/R22; existing g1
scenes already test [R21 and even e2](../alignment_factorized_constant_20261105/findings.md#existing-g1-scenes-second-component-null-checks).

Preparation17772071 verifies non-shear columns and source hashes against
the exact original-image replay. Eight pilot legs17772072 then gate
remaining72 legs17774673 (replaces cancelled pending array17772073). Images/measurement run in BlendEMU on
CPU; preserve original unseeded ngmix initialization and its known uncertainty.
Same C8 probabilities are recomputed at the new shear direction in17772094,
including a g1 cached-probability replay. Synthetic axis-index reducer check
17772095 passed known R12=.03,R22=.8,m2=3% with unequal leg counts.

Frozen retained and Cartesian seed611 GPU evaluations17772096 wait for
all images, C8, checks and seed-replication GPU work; final report17772097.
All stages are now complete; see [completed results](#completed40-scene-results).
No independent-population acceptance claim from these scenes.

## Runtime review and CPU parallelism

Owner questioned the35–55h estimate and whether Sersic rounding was retained.
The original scene writer rounds to122 logarithmic n nodes over[0.3,6.2]; new
g2 inputs copy the old catalogues changing only g1/g2. Renderer uses sersic_n,
not native unrounded n. Runtime audit17774601 checks all80 actual input arrays
against original g1 and this grid, plus rendered truth catalogues where available.
Audit completed successfully: all80 inputs are exactly equal to their original
g1 sersic_n arrays; all have122 distinct rounded values. Maximum numerical
distance from the reconstructed grid is8.9e-16; all eight rendered pilot
truth catalogues also pass. Receipt: [runtime_audit.json](runtime_audit.json).

Original constant job17643819_0 ran10 cases with10 MPI workers in4h37m:
render/detect1h36m, shape fitting3h01m. Original campaign used28 such constant
jobs (280workers), up to500 including training. New completed rendering stages
are50–62min/image, detection2–3min; no evidence of a rendering slowdown.
The long campaign estimate came from eight single-worker jobs and included
shape fitting. Raised pending remaining array17772073 to40workers on2026-11-06,
within cluster50-job/3500CPU/20000GB per-user limits, preserving the successful-
pilot prerequisite. No resubmission, science change or duplicate work.72remaining
legs need roughly two rather than nine batches at full allocation. Revised
provisional total is about12–18h remaining including evaluation, subject to
queue availability and first completed pilot shape timings.


## Owner-authorized CPU expansion

Owner authorized up to500 CPUs on2026-11-06. The unchanged pipeline has72
remaining independent case/leg tasks. New array17774673 packs these into nine
8-CPU/160GB allocations, permitting all72 workers concurrently within the
50-job limit; no within-image algorithm change. Keeps afterok pilot17772072.
Cancelled pending array17772073 only after rewiring evaluator17772096 to the
new array and existing seed-constant prerequisite17772054. Running pilot intact.
All72 indices8..79 covered exactly once; per-leg outputs/logs remain distinct.
Smoke17774672 completed in3s with all eight workers reporting MPI rank0,size1
and correct leg IDs. Initial smoke17774658 failed during environment activation
under shell nounset; removed nounset to match the original launcher, then passed.
At submission, production was pilot-gated; the completed timings are recorded
below. Earlier12–18h estimate assumed two waves; now one remaining image
wave is possible. Wait for completed pilot fitting timings before a firm ETA.
Existing bounded status17772183 reads the updated jobs.json automatically.


## Completed40-scene results

All eight pilot legs and nine packed jobs completed successfully. Packed image/
measurement allocations ran1h40m..3h37m, concurrently starting02:38:55 and all
finished06:15:30 on2026-11-06 (Europe/Berlin). Both-axis Cartesian/retained
responses and report17772097 completed; no pending work in this chain.

Main selection: independent per-leg measured FLUX_RADIUS>0.60 arcsec and
MAG_AUTO<25, no truth-property cuts; predicted C8 usability, generated selection.
m_j=100(Rsim_jj/Rmodel_jj-1); uncertainties are paired whole-case bootstrap
standard errors in percentage points, conditional on fixed models/catalogues.
[Conventions](../../CONVENTIONS.md#3-population-and-selection-order).

| Model | g1 m (%) | g2 m (%) |
|---|---:|---:|
| Retained GalSBI | +2.4288 ± 0.3102 | +0.1681 ± 0.2492 |
| Factorized Cartesian seed611 | -0.2739 ± 0.2995 | -0.5354 ± 0.2466 |

The old model's m2−m1 is−2.2607±0.4012pp; Cartesian gives−0.2615±0.3924pp.
For Cartesian, DeltaR12=−0.002959±0.002691 and DeltaR21=−0.000218±0.001936.
This supports improved directional consistency, not a robust0.3% calibration.
At magnitude<25.8,radius>=0.75 arcsec, Cartesian has m1=+2.3464±0.3713%
and m2=+2.3759±0.3365%. Thus the tighter-cut failure persists in both directions.
Full results: [full_report.json](full_report.json).

Matching receipts across80 measured g2 legs: zero crossmatched rows missing
shape records;114424 shape rows without crossmatches are excluded.47356388
parent rows absent from crossmatch are represented as unusable, never imputed
as valid shapes. No usable zero-shape rows. Corresponding g1 totals114259 and
47356465. Catalogue matching diagnostics are preserved per case/leg.

## Exact radius partition exposes cancellation

CPU17785999 partitions existing cached sums; no rerender or new model sampling.
First attempt17785969 read the original limited-cut cache and failed; corrected
to the pinned grid cache. All algebraic sum checks pass. Partition uses
MAG_AUTO<25.8 and radius>0.60 arcsec, separate simulation/model selection, own
full selected denominator per shear leg; contributions are absolute DeltaR.

| Measured-radius region | g1 contribution | g2 contribution |
|---|---:|---:|
| 0.60<r<0.75 arcsec | −0.011130 ±0.002126 | −0.011981 ±0.002297 |
| r>=0.75 arcsec | +0.009364 ±0.001373 | +0.009457 ±0.001202 |
| Sum | −0.001766 ±0.002204 | −0.002524 ±0.002178 |

The two contributions are correlated; total error is bootstrapped directly.
These are contributions to the broader selected sample, not separately
normalized bin biases. Tightening to0.75 removes the compensating negative
contribution and changes the denominator. The resulting tighter-sample errors
are+0.020848 (g1) and+0.020902 (g2).

A symmetric algebraic decomposition within radius/intrinsic-magnitude bins
assigns tighter-sample DeltaR11=+0.019569 to conditional measured-shape means
and+0.001279 to bin fractions; DeltaR22 gives+0.019600 and+0.001302.
This is an exact descriptive split, not a causal flow-versus-emulator budget.
The mean term includes differences within the coarse bins. True usability
still needs separate interpretation; measured selection on independent draws
is not conditional predictive closure.

Receipt: [nested_selection.json](nested_selection.json); finer radius scan and
intrinsic-size attribution in [new protocol](../alignment_radius_scan_20261106/rule.md#question-and-fixed-settings).

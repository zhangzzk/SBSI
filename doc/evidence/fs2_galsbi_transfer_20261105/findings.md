# FS2 to GalSBI: why the V3.9 result did not transfer

## Conclusion and scope

The FS2 success is a small **net joint error at one selection**, with sizable
opposite errors underneath it. The GalSBI retraining also has a directly
demonstrated conditional-flow failure near the measured-radius cut. Population
mix matters, but population mix alone does not explain the difference: within
the same intrinsic-size interval the measured shape–size relation changes,
and the GalSBI flow reproduces it substantially less accurately.

The band change also changed noise and PSF wings. Stamp geometry, training-label
history, and evaluation independence changed as well. The existing comparisons
do not identify a unique causal population-versus-band fraction. New matched
flow checks, a refinement ablation, PSF fits and a same-pixel stamp experiment
are completed here; no model was retrained or corrected.

## Definitions and uncertainty

Follow [project conventions](../../CONVENTIONS.md) and the
[predeclared protocol](rule.md#matched-standalone-diagnostic).
Absolute response error means **Delta R = R_simulation - R_model**;
m = 100 (R_simulation/R_model - 1), in percent. Uncertainty on m is in
percentage points. All radii below are measured FLUX_RADIUS unless explicitly
called intrinsic circularized Re. Pixel scale is 0.2 arcsec/pixel.

New standalone comparisons use each frozen F8 on its own harmonized 40-case
validation domain, matched usable g=0/g=0.05 random-direction pairs:
6,237,191 FS2 pairs and 5,817,708 GalSBI pairs. Selection is applied independently
to each leg's measured/generated outputs. The 2x2 forward influence response
uses each branch's own baseline mass and mean; measured-reference normalization
is also saved. This is a conditional-flow diagnostic, **not** the joint
constant-shear estimator or an additive component of its error budget.

The loader drops unmatched usable rows: FS2 46,186 on the zero leg and 43,095
on the sheared leg; GalSBI 35,353 and 32,816. No zero-shear pairs are excluded.
These losses are retained in the matching receipts; this matched-pair cohort
cannot measure the full detection/usable-event response.

Reported new response errors below are **R11**, not the earlier soft-cut
half-trace. Use 10,000 whole-case bootstrap draws, fixed models and two banks
of 16 antithetic draws with common random numbers between legs. These SEs
exclude retraining, unseen-population and shared-noise-realization uncertainty;
FS2 cases reuse a noise field, so case resampling is particularly not a full
uncertainty estimate. Stamp-test errors instead resample objects within fixed
images and bins, and have a narrower interpretation.

## What actually changed

| Property | FS2 retained V3.9 | GalSBI-dev retraining |
|---|---|---|
| Population / rendered photometry | FS2, LSST-r | GalSBI, LSST-i from its SEDs |
| Parent selection | r < 29 | HSC-i 14–28, then LSST-i rendering |
| Pixel noise rms | 0.3115 | 0.6159 |
| Moffat FWHM / beta | 0.73 arcsec / 2.224 | 0.71 arcsec / 2.458 |
| Galaxy fitting stamp | 48 pixels, subpixel-centre remeasurement | centred 65 pixels |
| Base flow shape labels | original mixed measurement history, later harmonized refinement | consistent centred measurements |
| Accepted harmonized refinement | 2 steps | 0 steps |
| Subsequent common-descent stage | retained step 0 | retained step 0 |
| Constant-shear galaxies / noise | reused training-family galaxies and shared noise | fresh cases 1000–1139, independent case noise |
| Evaluated role | primary half | all rendered galaxies |
| Sérsic support | about 0.5–6 | 0.3–6.2 |

Both pipelines implement circularized size, Re_major sqrt(q). GalSBI's retained
column/directory names containing `r` are historical interfaces: their values
come from LSST-i. This is not evidence that the old r-band flux was accidentally
used. Population density also increased (about 194 to 246 objects/arcmin²),
and size/Sérsic distributions changed. An identical numerical magnitude cut
therefore does not define an identical physical population or noise level.

History and source trails: sampler-recipe
[population comparison](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-03--fs2-vs-galsbi-population-corner-plots-input-and-detected),
[resimulation build](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-01--galsbi-re-simulation-build-lsst-i-table-scene-writer-blendemu-branch-10-case-pilot-launched),
and the [V3.9 artifact audit](../grid_regularization_history_20261105.md#v39-training-provenance-measured-cut-grid-and-retained-step-zero).

## Joint constant-shear history: the FS2 zero is selection dependent

| Per-leg measured selection | FS2 V3.9 joint m (%) | GalSBI-dev joint m (%) |
|---|---:|---:|
| Uncut | +2.09 ± 0.31 | +0.04 ± 0.17 |
| radius > 3 px, magnitude < 25.8 | +0.02077 ± 0.26515 | +2.74 ± 0.14 |
| radius > 3 px, magnitude < 25 | not evaluated here | +2.47 ± 0.14 |
| radius >= 3.75 px, magnitude < 25.8 | +1.14 ± 0.25 | +0.79 ± 0.18 |

The 25 rather than 25.8 test was indeed done. Its paired change is only
-0.27 ± 0.07 pp. It rules out the numerical magnitude limit as the sole
explanation; it does **not** rule out noise, PSF or band-dependent population
effects. The approximate equal-flux-S/N magnitude shift from the rms values
is 0.740 mag, but size, spectrum and blends prevent this from being an exact
depth match.

The earlier saved FS2 joint decomposition has +1.59 pp from true Re < 0.25
arcsec and about -1.55 pp from larger objects. This is direct evidence of
cancellation, not evidence that all FS2 components are accurate. See
[historical decomposition](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-03--why-fs2-passes-and-galsbi-fails-fs2s-zero-is-a-cancellation-and-the-3-px-size-cut-sits-on-the-galsbi-size-peak).

New [membership reduction](joint_membership.json) replays FS2's headline m
exactly. Of its 80 evaluation cases, 62 were F8 training cases and 18 F8
validation cases. Their main-cut m values are -0.194 ± 0.301% and
+0.762 ± 0.523%, respectively. This difference is too uncertain to identify
overfitting; neither subgroup is independent of the entire model-development
pipeline. C8 training/tuning groups give +0.004 ± 0.298% / +0.070 ± 0.578%.

## New matched standalone flow comparison

| Selection | FS2 Delta R11 ± case SE | GalSBI Delta R11 ± case SE |
|---|---:|---:|
| Uncut | -0.009082 ± 0.001066 | -0.006973 ± 0.002987 |
| magnitude < 25 | -0.022281 ± 0.001848 | -0.016490 ± 0.003485 |
| magnitude < 25.8 | -0.015445 ± 0.001448 | -0.006982 ± 0.003177 |
| radius > 3 px | -0.006984 ± 0.001396 | +0.013854 ± 0.003006 |
| radius > 3 px, magnitude < 25 | -0.021911 ± 0.001954 | +0.006948 ± 0.003359 |
| radius > 3 px, magnitude < 25.8 | **-0.013984 ± 0.001551** | **+0.013518 ± 0.003032** |
| radius >= 3.75 px, magnitude < 25.8 | -0.013613 ± 0.002374 | -0.018268 ± 0.004764 |

At the main 25.8 cut, FS2 simulation/model R11 are 0.470999/0.484983;
GalSBI 0.602779/0.589260. The flow residual changes sign between families.
Changing to measured-reference normalization preserves this sign difference.
The larger-radius GalSBI cut reverses the residual again: the failure is
localized in measured-size space, not a constant global response factor.

True circularized Re = 0.15–0.4 arcsec contributes -0.007416 ± 0.001422
in FS2 versus +0.014605 ± 0.003045 in GalSBI at the main cut. These are
additive contributions to the absolute standalone response error.

### Population-mix diagnostic

Within each family, standardize the 25 true-size/crowding cells to the other
family's matched-pair fractions and recompute the entire response estimator.
On common support, GalSBI main-cut m_R11 changes from +2.294% to +0.962%,
a paired -1.332 ± 0.222 pp change. At magnitude < 25 it changes from +0.971%
to -0.156%, a -1.127 ± 0.220 pp change. FS2 moves from -2.883% to -3.102%
at the main cut, -0.219 ± 0.173 pp.

Thus mixing populations can expose/cancel a residual, but exchanging these
two coarse distributions does not exchange the families' behavior. This is
not a causal fraction of the joint bias. Noise, Sérsic index, brightness,
within-cell size and neighbour structure remain unmatched. Weights are fixed
for the bootstrap. Common-support conditioning excludes just 0.0006253% of
FS2 pairs and no GalSBI pairs, reported explicitly in
[the paired reduction](refinement_comparison.json).

### Refinement ablation

Evaluate FS2's base guard and adopted checkpoint on identical data and random
draws. The two accepted refinement steps change model R11 by
-0.002009 ± 0.000033, moving m_R11 from -3.284% to -2.883%
(+0.401 ± 0.007 pp). This does not account for the approximately 0.0275
cross-family difference in absolute main-cut residuals. This ablation does
not undo FS2's earlier mixed-label training history.

## Direct conditional-density failure at the radius boundary

Measure baseline shape alignment as
A = sum(e_measured dot e_intrinsic) / sum(|e_intrinsic|²).
Each side bins its own outputs, with no artificial pairing of a generated
draw with one observed noisy measurement. Restrict true circularized
Re to 0.15–0.25 arcsec, then examine measured radius and magnitude.

At measured radius 2.7–2.8 px and magnitude < 25:

| Family | Simulation alignment | Flow alignment | Flow minus simulation |
|---|---:|---:|---:|
| FS2 | -0.525834 | -0.489735 | +0.036099 ± 0.004591 |
| GalSBI | -0.340546 | +0.041088 | **+0.381634 ± 0.004788** |

At magnitude < 25.8 the corresponding differences are +0.044738 ± 0.003500
and +0.144030 ± 0.002874. The GalSBI mismatch persists across the transition;
it is not only a count mismatch. The diagnostic bin is below the selection
boundary, where migration affects which objects enter the selected sample.
It is not itself the post-selection m or a conversion of alignment into bias.

![Baseline conditional shape-size comparison](shape_size_transfer.png)

The FS2 measurement valley and GalSBI sign change occur at different radii.
The FS2 flow tracks its valley much more faithfully. Holding true size within
one bin does not hold the PSF, noise, Sérsic distribution or blends fixed,
so this establishes a conditional-model error without assigning a unique
physical cause to the changed relation.

## Controlled stamp experiment and PSF audit

Refit the same original plus-leg images/detections in FS2 cases 40/80 and
GalSBI cases 1000/1080. Use true Re 0.15–0.25 arcsec, measured magnitude < 25,
radius 2.4–3.6 px; sample uniformly within each radius bin, up to 400/case/bin.
There are 16,727 objects and four fits per object (two stamps, two paired
optimizer seeds). No fitted-T cut is applied.

At radius 2.7–2.8 px, changing 48 to 65 pixels shifts alignment by
+0.00911 ± 0.00382 in FS2 and +0.02800 ± 0.01340 in GalSBI. These shifts
are small compared with the GalSBI flow mismatch above, and the sharp
cross-radius transition remains. At radius 3.0–3.1 px the shifts are
+0.02564 ± 0.01572 / -0.00964 ± 0.01222. Both cases are separately retained
in [stamp_comparison.json](stamp_comparison.json).

This argues against stamp size being the dominant explanation for this
local shape-profile difference. It does not test the original geometric-centre
labels, a complete rerendering, or the resulting constant-shear m. Sampling
and uncertainty are conditional on four fixed scenes; adjacent sources may
also share pixels. Original and refitted shapes are not bit-identical because
optimizer seeds differ from the production fits.

Negative fitted T is very common in this specifically targeted transition
region: at radius 2.7–2.8 px, about 95% of FS2 and 80% of GalSBI fits have T<0.
Those percentages must not be generalized to the full sample or compared
directly with the earlier uniform replay's 7.9%. Finite returned shapes span
negative/nonnegative-T outcomes, a plausible difficult distribution for the
flow. Descriptive branch contributions are saved; their prevalence alone
does not establish a signed shear bias or justify filtering them out.

The PSF change is larger than the FWHM numbers suggest. Applying the actual
Gaussian PSF fitter to the saved PSF stamps gives T_PSF = 0.493749 arcsec²
(FS2) versus 0.417795 arcsec² (GalSBI): **15.38% lower second-moment size**,
or about 8% lower linear size. Analytic Moffat half-light radii are 0.5268
and 0.4852 arcsec, so a fixed 0.6-arcsec cut has a different relation to the
PSF. These are distinct size definitions; the untruncated analytic Moffat
second moment must not be substituted for the fitted T. See
[PSF measurements](psf_comparison.json).

## What remains unresolved and the discriminating next experiment

The evidence supports three statements: FS2's net success contains
cancellation; GalSBI has a demonstrable flow shape–size error at the cut;
the change in the measurement distribution is more than changing counts of
small galaxies. It does not yet divide the error into population, noise,
PSF and training-history contributions.

Do not attribute the entire joint residual to the flow. Earlier four-scene
GalSBI checks retain an opposite-group response error of +0.007892 ± 0.001789
when the target group's shear is fixed, alongside a within-group error of
+0.013958 ± 0.004683. Those terms sum to +0.021850 ± 0.005689 but the
within-group term is not pure flow. See
[joint localization](../constant_split_20261105/findings.md).

The clean causal comparison crosses population with observing conditions:
FS2/GalSBI under r-like/i-like PSF and noise, using the same centred measurement
code, matched latent scenes and common random numbers between interventions.
Initially keep each galaxy's flux fixed to isolate PSF/noise; separately
change r/i SED-derived flux on the same GalSBI galaxies to isolate band
photometry. Within each cell compare noise and neighbour contributions,
shape–size/T-branch distributions, and both measured/model selected response.
Only then retrain the same recipe to determine whether correcting that
distribution restores calibration. Merely relabelling an i-magnitude column
as r or matching its limiting magnitude would not perform this experiment.

An untouched FS2 evaluation would separately test how much of the original
cancellation survives fresh galaxies/noise. It would not by itself isolate
the population-versus-band cause. No independent acceptance claim is made here.

## Reproduction and validation

External case moments and measurement rows:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/fs2_galsbi_transfer_20261105/`.
Scripts, compact reports, source/measurement receipts and jobs are retained
beside this document. Frozen model hashes are checked before evaluation.
No production artifact, preset or catalogue was changed.

The response algebra reproduces a known synthetic 2x2 matrix; disjoint
size/crowding contributions close on every bootstrap sample. The new measured
GalSBI hard-cut half-trace 0.7277161635668168 reproduces the earlier value
0.7277161635668159. The same-normalization model half-trace changes by
0.0002555 with a new draw stream; saved bank differences provide an integration
sensitivity check. R11 and half-trace, hard and soft cuts, and own versus
measured-reference normalization are kept distinct in the reports.

First GPU attempts failed on rare zero-radius log-profile underflow; v2
excludes undefined log-radius terms only, counts the draws, and leaves all
response sums unchanged. The initial stamp startup failed before fitting
because of shell environment activation, then all four repaired tasks passed.
The standardization reducer exposed an empty cell, now handled with explicit
common-support conditioning. Failures and successful job IDs are preserved
in [jobs.json](jobs.json); they are not silently omitted.

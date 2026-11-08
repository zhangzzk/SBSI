# Identical galaxies under FS2/GalSBI conditions, and controlled flow fits

## What is established

The owner's clarification is that the populations **should not** be too
different. A population difference alone is not an adequate explanation.
Holding the intrinsic galaxies exactly fixed, the PSF prescriptions alone
substantially change the measured alignment. This establishes an observing-
condition effect on the measurement distribution, not its contribution to
constant-shear calibration bias or the unique cause of density underfitting.

The earlier [learning audit](../alignment_learning_20261105/findings.md#frozen-checkpoint-result-density-underfitting-already-on-training-data)
found the GalSBI alignment failure already on training data, across Sérsic,
brightness, orientation and crowding strata. A different measurement
population can be harder to learn; it remains the flow's job to learn it.
The new controlled fits below test a representation change directly.

## Definitions

Use [population/selection conventions](../../CONVENTIONS.md#3-population-and-selection-order),
[shape conventions](../../CONVENTIONS.md#4-shapes-and-shear),
[response estimators](../../CONVENTIONS.md#6-response-estimators), and
[seed/model-selection conventions](../../CONVENTIONS.md#8-seeds-and-model-selection).
Standalone DeltaR11=R11_simulation−R11_flow uses matched usable forward
random-direction g0/g0.05 pairs and each branch's own normalization, with
independent measured/generated cuts per leg. These are not joint constant-
shear m estimates. Alignment errors have the opposite comparison order:
A_flow−A_simulation, with A defined below. The profile uses intrinsic
circularized Re[0.15,0.25)arcsec and each side's own measured/generated radius;
response totals cover the full matched validation population.

## Fixed-galaxy intervention: completed

5184 fits: 216 identical intrinsic contexts, two PSFs, three noise levels,
four paired noise/fitter seeds. All 5184 returned flags=0, finite g/T, |g|<1;
no exceptions. The grid equally weights circularized Re[0.15,0.20,0.25]arcsec,
n[1,2,4], intrinsic |g|[0.1,0.3,0.5], sky angle[0,45]degrees and
mag[23,24,25,25.8]. Results are descriptive grid summaries, without empirical
population weights or sampling-error bars.

Alignment A=sum(g_measured dot g_intrinsic)/sum(|g_intrinsic|²), consistent
with the earlier alignment panels. It is not a correlation coefficient,
response, m, or an additive contribution to m. Each row below holds the
intrinsic galaxy distribution fixed.

| Intrinsic circularized Re (arcsec) | FS2 PSF, no added noise | GalSBI PSF, no added noise | FS2 PSF, r noise | GalSBI PSF, i noise |
|---|---:|---:|---:|---:|
| 0.15 | -0.68750 | -0.83577 | -0.56951 | -0.38245 |
| 0.20 | -0.83170 | -0.10551 | -0.42402 | +0.03167 |
| 0.25 | +0.48361 | +1.23836 | +0.29092 | +0.55259 |
| Whole fixed grid | -0.34520 | +0.09902 | -0.23421 | +0.06727 |

Changing PSF alone with no added noise changes the sign of the alignment
projection in153/864 paired fits (17.71%); the count of T-sign changes is
also153/864. These are counts across four seeds, not864 distinct galaxies.
The negative-T fraction changes from79.63% to64.24%. Under r-like noise,
changing only PSF raises grid alignment by0.32788. Increasing noise at fixed
PSF changes alignment by+0.10343 for the FS2 PSF but-0.02640 for the GalSBI
PSF: there is no universal additive noise correction. Combined conditions
change grid alignment by+0.30148 and projection sign in210/864 paired fits.

PSFs are truncated Moffats: (FWHM,beta)=(0.73,2.224) and(0.71,2.458), truncation
4.5FWHM. Pixel scale0.2arcsec, magnitudes use zero point30, noise rms is
0/0.3115/0.6159. Rendering uses untruncated Sérsic profiles with circularized
half-light radius followed by area-preserving shear. Both use the same
centred65 galaxy measurement; the48px PSF stamp includes the renderer's
0.5pixel offset. The production one-Gaussian PSF and galaxy fitter is
unchanged except explicit paired RNG seeds. No added noise still uses the
production border-MAD weight estimate on the finite noiseless stamp.

**Limits:** isolated objects, fixed grid, zero applied lensing shear, no
SExtractor detection or FLUX_RADIUS measurement/selection. These numbers
cannot be substituted into the earlier radius-selected panel or converted
into a constant-shear m. They show that modest FWHM differences together with
PSF-wing changes can materially alter the fitted-shape transition. They do
not establish that either population is otherwise similar, that PSF alone
causes the flow miss, or that excluding negative-T fits would fix calibration.

Raw outputs and receipts: external `alignment_frame_20261105/images/`.
Figure: [identical_galaxies.png](identical_galaxies.png) (also PDF), generated with the Matplotlib skill and visually inspected. Compact results: [images_report.json](images_report.json) and per-context
[images_contexts.csv](images_contexts.csv). Image code is
`/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/audit_identical_galaxy_conditions.py`.
Image array17757925 completed all four tasks exit0,35–37seconds each.

## Gaussian-rendering control: completed

A second5184-fit control changes the rendered PSF family to Gaussian, with
intrinsic T=0.4937487425/0.4177950357arcsec² (the previously audited fitted
production PSF sizes). It keeps the galaxies, pixel integration, fitter,
stamps and paired noise/seeds fixed. This matches the fitter's PSF family,
not the original Moffat FWHM or detailed images.

With no added noise, the negative-T fraction is **zero for both widths**,
and alignment is+0.88003/+0.87277, versus-0.34520/+0.09902 for Moffats.
Under production r/i noise respectively, negative-T is1.74%/4.28%, versus
77.66%/58.10% for Moffats on this targeted small-galaxy grid. All Gaussian
control fits succeeded. This strongly implicates PSF-model mismatch in this
measurement pathology; the intervention changes the observation, so the
same-Moffat-image three-component fitter control is the sharper next test.

Gaussian array17758019 completed all tasks exit0. Compact results are
[gaussian_report.json](gaussian_report.json) and
[gaussian_contexts.csv](gaussian_contexts.csv); raw external folder is
`alignment_frame_20261105/images_gaussian/`. A predeclared same-image test
17758135 replaces only diagnostic PSF ngauss1 with ngauss3. Production code
and catalogues remain unchanged.

## Same-image PSF-component control: completed

Array17758135 repeats the **same Moffat images** with only the diagnostic PSF
mixture changed from one to three Gaussians. The galaxy Gaussian model, priors,
noise and initial RNG seeds are unchanged; changing PSF mixture dimension can
change subsequent optimizer random draws. Compare identical successful rows
on both sides, rather than interpreting changes in successful-row selection
as shape improvements.

| No added noise, identical successful rows | FS2 PSF | GalSBI PSF |
|---|---:|---:|
| Paired usable fits / attempted | 719 / 864 | 703 / 864 |
| Alignment, 1-Gaussian PSF fit | -0.36190 | +0.07033 |
| Alignment, 3-Gaussian PSF fit | +1.03593 | +0.97009 |
| Negative T, 1-Gaussian PSF fit | 79.83% | 64.58% |
| Negative T, 3-Gaussian PSF fit | 0% | 0% |

With production noise on the same successful rows, negative T drops from
78.44% to0.70% for FS2/r-noise and57.47% to3.41% for GalSBI/i-noise.
This directly implicates the one-Gaussian approximation to the Moffat PSF
in the pathological negative-size/shape behavior of this targeted grid.
It does not identify the fraction of the real selected constant-shear bias
or show that original FS2/GalSBI flow differences have only one cause.

**Not a deployable fix:** 918/5184 fits (17.71%) raise `BootPSFFailure: no good
psf fits` with the existing ntry2 initialization; these failures repeat across
noise levels because PSF images/seeds are fixed. Retain every failure in raw
outputs; the table conditions on success in both methods. A robust PSF fitter,
new measurement labels and full retraining/selection/calibration checks would
be needed before changing production. No production measurement was modified.

All four tasks completed exit0 in120–148seconds. See
[paired_psf_components.json](paired_psf_components.json),
[psf_components.png](psf_components.png) (also PDF; visually inspected),
[moffat3g_report.json](moffat3g_report.json), and
[compare_psf_components.py](compare_psf_components.py).

## Controlled flow improvement: completed, no accepted replacement

[Predeclared rule](rule.md#cartesian-versus-intrinsic-frame-density) compares
fresh Cartesian and intrinsic-axis shape outputs on the same40 training cases,
all usable measurements of both legs, eight original physical inputs, common
architecture/initialization/batches and120epochs×2million rows. Initial
learning rate3e-5, identical plateau schedule, best checkpoint by fixed500k
validation NLL alone. No response/alignment penalty or measured training cut.
The coordinate change is invertible and determinant1; original orientation
inputs remain present. Custom-format checkpoints cannot be silently loaded
as ordinary world-frame flows. Retained presets are unchanged.

This is a matched representation experiment, not an equal-budget rerun of
original160-case training. Failure of both short fresh fits would not prove
that the architecture cannot learn the relation. Improvement must include
validation density, selected-bin alignment/counts and neighboring-cut response,
not just a single alignment point. Even success here is not joint constant-
shear acceptance.

Training17756866 completed both tasks: Cartesian480.6seconds, frame527.2seconds,
both selected epoch119. All40 validation cases completed in171.6/176.7seconds;
reducer17757938 completed in20seconds, all jobs exit0. Same500k NLL rows and
all40-case response/profile cohort; two16-draw antithetic banks;10000 paired
whole-case bootstrap resamples. Error bars below are case SEs conditional on
fixed fitted models; they exclude retraining/model-selection uncertainty.

| Model | Validation NLL, lower is better | Alignment error at 2.7–2.8px, mag<25 | Delta R11, radius>3 & mag<25 | Delta R11, radius>3 & mag<25.8 |
|---|---:|---:|---:|---:|
| Retained GalSBI | -3.693633 | +0.38163 ±0.00479 | +0.00695 ±0.00332 | +0.01352 ±0.00298 |
| Fresh Cartesian | -3.531801 | +0.31488 ±0.00471 | +0.01588 ±0.00334 | +0.02552 ±0.00300 |
| Fresh intrinsic frame | -3.545000 | +0.25007 ±0.00486 | +0.02527 ±0.00333 | +0.02753 ±0.00300 |

The matched frame-minus-Cartesian alignment change is-0.06481±0.00119 at
mag<25 and-0.02026±0.00057 atmag<25.8. This shows that representation can
improve this local relation. It does not repair the full calibration: frame
response is worse at the main cut and both fresh density fits lose to the
retained flow. At radius>=3.75/mag<25.8, DeltaR11 remains negative:
retained-0.01827±0.00473, Cartesian-0.02297±0.00478,
frame-0.02121±0.00474. At2.7–2.8px/mag<25, measured count52985 versus
retained52729.6/Cartesian52297.1/frame53524.4; atmag<25.8 frame overpopulates
the bin134004.9 versus129214 measured. No response offset was applied.

Both fits continue improving near the budget endpoint. Cartesian also has a
validation spike to-2.18029 at epoch83; the frame scheduler halves its rate
at epoch117. This is not evidence of completed convergence or a decisive
architecture-capacity limit. The original NLL-only checkpoint has NLL-3.694125
on the same rows. Its stronger density fit and larger original training budget
remain relevant when comparing to the fresh pilots.

**Decision:** retain the current models. No joint constant-shear test of these
candidates was launched because they fail the predeclared density/response
requirements. The local alignment statistic is informative but is not a
sufficient model-selection criterion. See [comparison.json](comparison.json)
and [flow_comparison.png](flow_comparison.png) (also PDF).

The scheduled status-only job17757962 was cancelled after all stages finished;
a final read-only status receipt was saved immediately. No science jobs remain
running from this experiment.

## Validation and next interpretation

Rotation/inverse/norm, standardized Jacobian determinant including zero-shape
fallback, physical support, density change of variables and exact reload
passed; density identity maximum error5.0821e-7 (tolerance3e-5), saved in
[verification.json](verification.json). All experiment Python sources parse
and all experiment launchers pass `bash -n`. The inherited response reducer has a
known-matrix/additive-cell check. Actual measured replay differs by at most
6.37e-12 in saved moment sums. The largest model-bank R11 difference is0.00108
at the tighter neighboring cut, below its case SE0.00474; main-cut bank
differences are at most0.000243. Image IDs cover216 unique contexts in each
intervention, with every attempt/failure accounted for. All figures were
visually inspected.

Next resolve PSF-fit convergence and test the measurement-model intervention
on actual selected scenes, with matched detections and measured cuts. Any
new measurement convention requires new flow/emulator labels and full joint
constant-shear validation. For the existing labels, longer controlled density
fitting or a more flexible distribution remains open; the frame pilot alone
is insufficient. These are next investigations, not completed improvements.

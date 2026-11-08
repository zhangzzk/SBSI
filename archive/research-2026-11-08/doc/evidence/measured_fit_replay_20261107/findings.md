# Controlled measurement replay

## Completed four-scene result

Changing only the optimizer seed changes at least one shape component by>0.1
in1,042 of8,189 valid paired fits (12.72%). Tightening ftol/xtol from1e-5 to1e-10
at fixed seed changes a component by>0.1 in323 of8,165 valid pairs (3.96%).
These are differences on exactly the same images, cutouts, detected positions,
PSF stamps, prior and weight-estimation procedure. All returned valid shapes
have fit status zero. The original seed was unrecorded; default seed A versus
historical fits yields a similar12.71% rate.

The sample contains2,048 common-selected IDs from four prechosen scenes,
each measured in four shear exposures. The three interventions require24,576
fit calls plus exact-repeat gates. Thirty calls fail to return a shape;
2,036 IDs remain valid across all three modes and four exposures. The output
uses−999 for missing flags, not an observed ngmix flag value; exceptions are
preserved in each case receipt. The standard catalogue membership was not
changed and no production outputs were overwritten.

## What this says about the directional gap

The small replay subset cannot establish how much of the full40-scene
R11−R22=0.00965 is caused by optimizer behavior. On its2,036 common valid IDs,
tight-minus-default changes the gap by−0.05547±0.05378 (conditional row error),
or±0.04119 with the unstable four-scene jackknife. Historical and new response
estimates themselves have large sampling errors. Do not claim this repairs or
explains the full-population directional difference.

Pairing optimizer seeds across shear legs reduces the sampling variation in
this pilot, but fixing seeds alone does not demonstrate correctness or remove
possible systematic fitting error. This pilot establishes a measurement
repeatability problem that needs resolution before interpreting sub-percent
response differences as model-only deficiencies.

## Why success flags do not prevent the problem

Production ngmix uses `ftol=xtol=1e-5`, `maxfev=4000`, and `Runner(ntry=2)`.
Installed `ngmix.runners.run_fitter` exits at the first result with flags zero;
`ntry=2` retries failures rather than comparing two successful solutions.
Thus existing success flags do not test stability with initialization or
whether a materially better solution exists. The continuation diagnostic
compares the objective from the returned solution at stricter convergence.

## Isolated orientation controls

Eight noiseless orientation rings cross Sersic n=1/4, circularized radius
0.35/0.7arcsec and production Moffat/Gaussian PSFs. Each averages16 angles and
four symmetric subpixel phases. Their largest absolute R11−R22 is0.0002923,
well below the measured common-sample0.00863. These controls lack noise,
neighbours and detection and cannot rule out those effects. They show that
these simple isolated pixelized profiles do not reproduce the observed gap.
All ring fits return success. `orientation_ring.json` and raw arrays remain
in the project cache.

## Validation and limits

Job17821683 completes0:0 in9m42s after the first-scene validity gate passes;
17821716 completes0:0 in1m34s. Input matching is one-to-one, fixed sample IDs
and image/PSF paths are retained, and each case's first fit repeats exactly
within its job. Four data hashes validate before reporting. Raw and generated
fitter source hashes are retained. Measurement scripts live in BlendEMU and
analysis in SBSI; no production fitter or model changes.

A later cross-node replay failed, including a0.1273 difference at the same
nominal seed for a difficult fit. It is not validated across architectures.
The continuation test therefore uses an explicit within-job exact-repeat
control and records the cross-node discrepancy. See the
[adaptive protocol](rule.md#adaptive-optimizer-continuation-diagnostic).

## Continuation identifies the galaxy optimizer

On the enriched32 seed-sensitive plus32 stable-control images, live default
fits repeat exactly within the same allocation. The PSF fits from seeds A/B
differ by at most5.6e-17 in their Gaussian-mixture parameters. The large shape
changes therefore arise in the galaxy fit, not stochastic PSF fitting.

Among the32 deliberately seed-sensitive cases, tightening convergence reduces
seed-pair component differences>0.1 from32/32 to19/32. Continuing the original
solutions with the PSF fixed also leaves19/31 valid pairs above0.1. Some tighter
continuations markedly lower the objective; thus early stopping contributes,
but simply reducing the tolerance does not resolve the competing solutions.
Among32 stable controls, median A/B shape difference decreases from0.000273
to0.00000152 under tighter fitting. All comparisons retain invalid/missing
results explicitly; one continuation fails to return a shape.

CPU17821777 completes the continuation diagnostic after earlier diagnostic
attempts exposed cross-node replay differences and a missing-shape result that
needed explicit recording. `stability_report.json` contains paired valid counts,
objective changes, flags and replay limits. These selected32/32 counts are
mechanism tests and must not be read as population frequencies.

## Candidate multistart check

Job17821808 completes two independent banks of16 tight fits on each of the64 images,
freezing the PSF and choosing the successful solution with lowest residual-
square objective. Compare bank sizes1/2/4/8/16. This tests whether explicitly
comparing solutions improves repeatability. It changes the measurement rule
and is not a calibration repair until tested on an appropriate population.
No production changes or catalogue replacement are made.

In the32 seed-sensitive examples, independent best-of16 banks disagree by>0.1
in1/32, versus6/32 for the first start of each new bank. However,2/32 previously
stable controls acquire differences>0.1 at best-of16 as better competing
solutions are discovered. Therefore multistart is informative but not a
validated replacement. These different initialization streams must not be
compared directly with the original tight-A/B19/32 as a population improvement
rate. The full-population next test preserves the original measurement rule
and pairs optimizer seeds to assess the directional gap with adequate
precision: [focused repeat](../measured_seed_response_20261107/rule.md#purpose).

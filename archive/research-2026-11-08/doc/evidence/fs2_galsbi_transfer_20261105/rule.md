# FS2 to GalSBI transfer audit, 2026-11-05

## Question and fixed scope

Owner requests a deep investigation of why the V3.9 recipe works on FS2 but
its GalSBI retraining does not. Population, photometric band and magnitude
selection are the initial hypotheses. Audit also records measurement-stamp,
PSF/noise, scene/noise independence, training-label and parent-role changes.
Retain all model identities; no training, offsets or empirical calibration.

## Matched standalone diagnostic

Evaluate the retained FS2 F8 and GalSBI-dev F8 on their respective harmonized
40-case validation domains. Use matched usable g=0/0.05 pairs for this diagnostic;
this is a conditional flow check, not the full joint constant-shear estimator.
Frozen installed F8 runtime and its case loader, target transform and physical
draw method are used. Report dropped/unmatched pairs from the loader.

Hard measured cuts: uncut; magnitude<25; magnitude<25.8; radius>3 pixels;
radius>3 and magnitude<25; radius>3 and magnitude<25.8; radius>=3.75 and
magnitude<25.8. Each leg uses its own outputs. Include no artificial object
pairing across simulated and generated measurements. Responses use the same
random-direction forward design on both sides. Save baseline selected mass
and shape, shape/shear and mass/shear cross moments, and the shear design.
Report both measured-reference and each-model-own-baseline normalization.

Two independent banks of 16 antithetic draws, with CRN between shear legs
within each bank, provide an integration sensitivity check. Seed is
202611050700 + 10000019*case + 1000003*bank + block; block size 2048.
Use all validation pairs. Sampling uncertainties resample entire cases;
they do not include model training or unseen population uncertainty.

Fixed true circularized-size edges: 0.15,0.25,0.4,0.7 arcsec. Fixed physical
crowding edges (nbr_close_100): 0.1,0.3,1,3. Save the joint 25-cell moments,
including per-cell shear normal matrices, so counterfactual weighting can
recompute the estimator. Reweighting is a distribution-standardization
diagnostic, never a calibration correction or proof of a causal population
effect. Absolute magnitudes in different bands are not equated.

At g=0, also save count, intrinsic ellipticity power, dot product of measured
and intrinsic shapes, log measured radius, their cross moment and squares,
in true-size x measured-radius bins (2–5 pixels at 0.1-pixel spacing, with
open ends), at magnitude<25 and <25.8. These summaries test the shape-size
alignment profile without assigning a generated draw to an observed galaxy.

## Execution and gates

Two GPU workers maximum, one per family, through Slurm. Source hashes and
model hashes recorded. No archived launcher is restarted. Stop on identity
or nonfinite-output errors. Validate disjoint-bin additivity and response
algebra with synthetic examples; compare totals against existing checks
where the cohort/cuts agree. Preserve differences in draw streams explicitly.

CPU-only saved-result reductions and provenance inspections may run alongside
the GPU jobs. Later controlled image or held-out-scene tests need an explicit
extension here before results are inspected.

## Refinement ablation, declared after the first comparison

The first matched check finds FS2 standalone R11 overprediction and GalSBI
underprediction at the main cut. Evaluate the FS2 pre-refinement base guard
checkpoint on exactly the same harmonized validation rows, cuts and both draw
banks. This isolates the two accepted FS2 response-refinement updates from
the fixed-data comparison; use paired case resamples. It is not retraining or
an end-to-end constant-shear result. The GalSBI retained weights already equal
its base guard, so no duplicate GalSBI run is necessary.

The initial profile summary encountered zero-radius floating-point underflow
in a few extreme generated draws. Version 2 excludes only undefined log-radius
profile entries and counts them explicitly; all response sums remain unchanged.
Both complete family runs use version 2, with failed first attempts preserved.

## Stamp-only measurement experiment

The baseline alignment profiles differ strongly between FS2 and GalSBI.
To isolate one measurement change, use original plus-leg pixels and detections
of FS2 cases 40,80 and GalSBI cases 1000,1080. Select matched detections with
true circularized Re in [0.15,0.25), MAG_AUTO<25 and measured radius in
[2.4,3.6) pixels, without selecting on fitted shape or T. Sample at most 400
IDs uniformly per 0.1-pixel radius bin, seed [20261105,case,700].

Fit each source with 48-pixel historical cutout plus its exact subpixel
Jacobian centre, and with the centred 65-pixel cutout. Keep the original
science/PSF pixels and detections. Use the same explicit fitter seed across
stamp variants, and two independent seed replicates per source. Save all
exceptions, fitted shapes, T, flags and likelihood. Do not introduce a T cut.
Production files remain read-only. Measurement code lives in BlendEMU.

Report the paired change in shape–intrinsic alignment and negative-T fraction
by measured radius, with conditional object-bootstrap uncertainty and the
two cases separately. This is a mechanism diagnostic on fixed images, not a
new full-sample shear calibration or a test of the band/population alone.

## Reduction clarifications and exploratory follow-up

One of the 25 size/crowding cells is absent from GalSBI and contains
0.0006253% of FS2 matched pairs. Standardization contrasts in
`refinement_comparison.json` explicitly condition both distributions on their
common support and report each excluded mass. The earlier full reduction
records its unsupported target mass; do not interpret it as extrapolation.

After viewing the completed stamp comparison, add the negative-T and
nonnegative-T contributions to alignment in each radius bin. Their sum must
equal the original full alignment. This is an exploratory description of
the fitted-shape transition, not a new shear-bias estimate or a T-cut proposal.

# Small circularized-radius response diagnosis

## Scope and verification — 2026-09-12

The owner asked why the smallest circularized-radius feature bin has poor
predicted responses, and whether an observational size cut should use measured
circularized size. This diagnostic reuses the full1,708,854 matched validation
pairs/40cases and all16 saved64-draw groups for each selected flow: covariance
grid216 lambda1 epoch146, identity/no-covariance grid216 lambda1 epoch182 and
the prior grid-free shape+radius lambda10 epoch154. There is no new training,
selection threshold optimization, detector weighting, R_blend or inference.
These are development-validation scenes, not an independent new test sample.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_small_circularized_radius_diagnosis_v1`.
Frozen manifest, common.py, diagnose.py, launcher and worker hashes bind the
original arrays and48 saved response groups. CPU Slurm16467838 completed129s,
exit0/empty stderr. Exact pair indices/identities reproduce; decoding the
original condition and reconstructing a0 sqrt(q0) agrees with the invariant
circle cache to maximum1.21e-7arcsec. All original eight circularized-bin
responses and paired case uncertainties reproduce the previous feature plot.

The separate frozen nll_context_v1 stage, GPU Slurm16467864, completed21s,
exit0/empty stderr. It re-evaluates all3,455,704 validation NLL rows for all
three checkpoints, reproducing each previous full NLL. Its independent
review.py solves the ten diagnostic designs from saved sufficient statistics;
predictions, residuals,40-case jackknife SE and16-group integration SE agree
to5.49e-16. results.json, sufficient.npz, pair_features.npz, completed.json,
nll_context_v1/results.json and final_review.json preserve the evidence.

Responses use the original unit-pair forward0-to0.05 WLS convention, fitting
within each subgroup; shape summary is (R11+R22)/2. Radius responses are
physical measured FLUX_RADIUS in pixels/shear,0.2arcsec/pixel, projected in
the original intrinsic frame. They are not equal-cell grid means or actual
SBSI inference m. See [response conventions](CONVENTIONS.md#6c-extraction-convention-must-match-on-both-sides).
Subgroups with fewer than1000pairs or singular full/deleted-case designs are
reported and omitted from response fitting. All population counts are retained.

## Size definitions and the actual observational cut

Three quantities must be distinguished:

| Quantity | Definition | Role here |
|---|---|---|
| True semi-major radius a0 | Baseline catalogue Re | Fixed physical support0.37<a0<1.5arcsec |
| True circularized radius | a0 sqrt(q0), both quantities pre-shear | Invariant flow condition and horizontal plot axis |
| Measured circular-aperture radius | SExtractor FLUX_RADIUS at PHOT_FLUXFRAC0.5 | Per-leg measured selection, >=3.75pixels=0.75arcsec |

The simulation configuration references MultiBand_ImSim/config/kids_sims.sex
and sex_image.param, the latter requesting FLUX_RADIUS. The former and the
SExtractor command construction in modules/Astromatic.py do not override
PHOT_FLUXFRAC. Retained producer logs for source cases40,100 and119 record
SExtractor2.29.0, and the pre-training RoyBranch2.29.0 executable's `-dd`
output gives its default0.5 (the available2.28 executable agrees). Pixel
scale0.2 is explicit in the simulation configuration and passed to SExtractor.
Local source/config paths and their hashes are saved in
measured_size_definition.json; the old record inspected configuration
references and defaults, not a recovered historical executable hash.

FLUX_RADIUS is the radius of a circle enclosing the requested fraction of
FLUX_AUTO; it is already a circular-aperture size, not a semi-major size to
multiply by another sqrt(q). It is also not identical to the area-equivalent
radius of a fitted ellipse a sqrt(q), or to a PSF-corrected intrinsic radius.
See [the official SEP equivalence documentation](https://sep.readthedocs.io/en/stable/apertures.html#equivalent-of-flux-radius-in-source-extractor)
and [SExtractor's circular growth-curve implementation](https://github.com/astromatic/sextractor/blob/master/src/growth.c).

The ConstGold selection tests already cut this measured quantity, separately
in each leg and on each model's own simulated measurements. Their fixed truth
support remains the original semi-major domain. The response feature plot
uses no measured selection. Changing the truth-domain coordinate or adding a
true circularized cut is a different population restriction; it does not
implement an observational measured-size cut.

## Size-bin discrepancy and density fit

The first plotted circularized bin is0.089619–0.286003arcsec and contains
213,607pairs, exactly12.5% of validation. Measured shape half-trace is
0.717491 ±0.012363(case); measured radius parallel is0.815488 ±0.014629.
The table gives model-minus-measured residuals with paired case SE:

| Model | Shape prediction | Shape residual | Radius parallel residual | NLL on431,772 individual rows in this size band |
|---|---:|---:|---:|---:|
| Grid-free lambda10 | 0.583347 | -0.134144 ±0.012462 | +0.051815 ±0.014714 | -1.763016 |
| Grid216 lambda1 covariance | 0.609610 | -0.107881 ±0.012451 | +0.031577 ±0.014660 | -1.715879 |
| Grid216 lambda1 no covariance | 0.589297 | -0.128194 ±0.012477 | +0.047196 ±0.014658 | -1.771950 |

Shape integration SE is only about4e-5; this is not latent Monte Carlo noise.
The radius problem is weaker than the shape problem here, and its largest
absolute marginal residual is not uniquely in this first bin. Neither radius
parallel nor shape has the original semimajor-condition radius sign failure.

No-covariance improves same-row NLL by0.008933nat/row versus the old model,
while leaving most of the shape-response deficit. Covariance improves the
shape residual by0.026263 but worsens NLL by0.047137nat/row. Absolute NLLs
across different size populations cannot themselves diagnose worse fit:
conditional entropy and density coordinates also differ. These model contrasts
compare exactly the same rows and common density convention.

The deficit extends through well-populated smaller-size intervals. For the
covariance flow the shape residual is-0.1851 ±0.0464 at0.20–0.225arcsec,
-0.1735 ±0.0222 at0.225–0.25, -0.0937 ±0.0249 at0.25–0.27 and
-0.0529 ±0.0227 at0.27–0.2860. It is+0.0052 ±0.0156 at0.2860–0.30.
The484pairs below0.15 are too few for the preset minimum-count gate; the
problem is not attributable only to that extreme tail.

## Composition and what simple explanations fail

There are1,736,196 original training NLL rows below the first validation-bin
boundary,12.561% of13,821,950rows. The bin is not globally rare or outside the
trained circularized range. This does not establish dense coverage in every
combination of conditions or of perturbation directions.

The semi-major lower bound does induce a curved joint support. For the shape
convention |e|=(1-q)/(1+q), a0>0.37 implies, at fixed circularized r_c,

    |e0| > max(0, (0.37^2-r_c^2)/(0.37^2+r_c^2)).

At r_c=0.25arcsec the minimum is about0.373. In the first bin, median
a0=0.41321arcsec, median|e0|=0.45932, and73.44% have a0<0.45.
The next bin's medians are0.44318arcsec and0.36495. Thus a marginal
circularized-size bin is also a different intrinsic-shape population.

This is not evidence that crossing the baseline support edge causes the
whole deficit. Only7.70% of first-bin pairs have post-shear implied
a1=r_c sqrt((1+|e1|)/(1-|e1|)) below0.37. The92.30% not crossing have
covariance shape residual-0.12025 ±0.01378; crossing pairs have
+0.04027 ±0.05571. Such implied-a crossings are not a full NLL-support test:
the sheared rows are themselves part of the training construction. For
first-bin objects0.10–0.20 above the intrinsic-|e| support boundary, the
residual is still-0.10830 ±0.02633.

The discrepancy is also not explained simply by an excess of faint objects
or a single crowded slice. Covariance shape residuals within the first bin
are-0.3228 ±0.0693 at true magnitude22–24, -0.1530 ±0.0219 at24–25 and
-0.0382 ±0.0115 at25–25.8. It persists across the reported Sersic and
intrinsic-|e| intervals. Near-neighbour-flux medians are0.7071 and0.7039 in
the first two circularized bins. The lowest near-flux subgroup still has
residual-0.1132 ±0.0298. Omitted scene features and interactions remain
possible; these are marginal subgroup controls, not a causal regression.

The grid's first circularized interval spans0.086924–0.296481arcsec, and
intrinsic|e| is not a grid axis. Its constraints do not separately require
correct responses in these finer size/shape/output subsets. The grid therefore
does not guarantee resolution of this conditional discrepancy. The evidence
does not uniquely identify finite model capacity, the optimization tradeoff,
omitted state, or the paired-survivor versus individual-row NLL populations.
Even a small density loss change need not tightly constrain a local shear
derivative. No conclusion of optimizer convergence is warranted here.

## Relation to the measured-size selection

First-bin median measured radius is0.73280arcsec before shear and0.73360
after shear. Only4 and6 of213,607 observations respectively are below
0.30arcsec: a measured0.30arcsec cut would retain essentially all of this
intrinsically small bin. It is not interchangeable with a true0.30arcsec cut.
The existing0.75arcsec measured cut keeps44.63% and44.78% per leg;
92,062pairs (43.10%) pass in both. The full shape/magnitude/radius cuts keep
28.08% and27.90% per leg, on the original paired-validation universe.

For a diagnostic split defined by the actual observed radii in both legs:

| Observed-radius subset within first bin | Pairs | Covariance shape residual | No-covariance shape residual | Covariance radius residual |
|---|---:|---:|---:|---:|
| All | 213,607 | -0.107881 ±0.012451 | -0.128194 ±0.012477 | +0.031577 ±0.014660 |
| Fails0.75arcsec in at least one leg | 121,545 | -0.191506 ±0.021744 | -0.236766 ±0.021736 | +0.120451 ±0.007259 |
| Passes0.75arcsec in both legs | 92,062 | +0.002361 ±0.012611 | +0.014990 ±0.012573 | -0.086263 ±0.033882 |

The observed small-radius failures carry the large negative shape discrepancy
in this partition; the radius residual changes sign. This associates the
problem with the joint measured shape/size distribution and cautions against
interpreting accurate marginal radius response as adequate selection modeling.

Crucially, this table uses an observed-output mask to choose object identities,
then evaluates the original uncut model means at those objects. Conditioning
on noisy observed radii can itself change measured conditional responses.
It does not apply the same radius restriction to model draws and is not a
model-selected response, selection calibration test or m. Agreement in its
pass subset therefore does not prove that the actual measured-radius cut fixes
the bias. Selection depends on joint shape/size probabilities near the cut,
their shear changes and changing denominators; marginal mean responses alone
cannot establish it. The separately authorized new-flow training-scene
selection comparison applies each leg's own measurement cuts and remains the
appropriate ongoing comparison. No new cut or correction is adopted here.

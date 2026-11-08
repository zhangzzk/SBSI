# Why intrinsically small galaxies measured large carry the response discrepancy

## Confirmation and scope

Confirmed on the existing four GalSBI constant-shear cases, with GalSBI-dev
unchanged. Intrinsic circularized Re 0.15–0.4 arcsec contributes
**+0.021526 ±0.007451** of the total **+0.021850 ±0.005689** R11 discrepancy
after FLUX_RADIUS >3 pixels (0.6 arcsec) and MAG_AUTO <25. This is almost all
of the point residual, not a precise causal percentage. These galaxies are
75.77% of the selected case/leg rows, so they are a large population.

The prior magnitude-only comparison had error −0.000253 ±0.004458; adding
the size cut exposes the discrepancy. See the
[original localization](../joint_residual_localization_20261105/findings.md#where-the-original-joint-residual-sits).

All errors below are **simulation minus model in absolute response units**,
not percent m or response errors normalized within each bin. For definitions,
see [response](../../CONVENTIONS.md#6-response-estimators) and
[multiplicative bias](../../CONVENTIONS.md#7-multiplicative-bias).

## Small intrinsic and large measured does not itself establish inflation

Intrinsic circularized Re excludes the PSF. SExtractor FLUX_RADIUS is measured
on the convolved image and uses a different flux/size definition. The actual
circular Moffat PSF has FWHM 0.71 arcsec, beta 2.45832751984934 and truncation
at 4.5 FWHM. Its analytic half-light radius is approximately **0.482 arcsec,
or 2.41 pixels**, before pixel convolution. Thus the 3-pixel cut is already
close to the PSF's own size.

This makes PSF broadening an essential baseline before assigning excess size
to noise or blending. The analytic PSF radius is not an individual galaxy's
noiseless SExtractor radius, and radii cannot simply be subtracted in
quadrature for these profiles and measurements. No fraction of selected
objects is classified as noise-inflated from this comparison.

## New crowding localization

For the same intrinsic-small population, split the existing crowding feature
more finely. It is log1p of a neighbour/target flux ratio with a 1-arcsec
Gaussian distance weight. It is not the actual contaminating flux fraction
inside the photometric aperture; even the lowest bin is not certified isolation.

| Crowding | Share of all selected rows | Total error contribution | Opposite-group error contribution |
|---|---:|---:|---:|
| <0.01 | 19.07% | −0.004531 ±0.002981 | −0.003119 ±0.001351 |
| 0.01–0.1 | 28.88% | +0.021705 ±0.002470 | +0.004545 ±0.000799 |
| 0.1–0.3 | 16.66% | +0.005959 ±0.001630 | +0.004446 ±0.001503 |
| ≥0.3 | 11.16% | −0.001607 ±0.002773 | +0.000418 ±0.001865 |

The positive residual peaks at modest crowding. The earlier <0.1 population
combines a negative contribution below 0.01 and a large positive one at
0.01–0.1. Calling that earlier population isolated would obscure this.
The most crowded group also does not carry the positive error despite much
larger measured sizes and flux excesses.

For the four crowding bins, mean selected radius is respectively
3.51, 3.74, 4.11 and 4.35 pixels; mean measured/truth flux ratio is
1.07, 1.21, 1.48 and 2.45. The fractions with measured flux >1.5 times the
target's truth flux are 3.73%, 7.07%, 27.87% and 71.23%. These trends are
consistent with neighbour contamination, but magnitude and signal/noise
also change across bins. They do not make crowding a causal error attribution.

## Magnitude and measurement quality

The new fixed-truth-magnitude split, within intrinsic Re 0.15–0.4 arcsec, is:

| Intrinsic magnitude | Share of all selected rows | Total error contribution | Selected fraction with photometric S/N <20 |
|---|---:|---:|---:|
| <23 | 10.23% | −0.001398 ±0.000812 | 0.00% |
| 23–24 | 18.02% | +0.002607 ±0.001145 | 1.04% |
| 24–25.5 | 44.38% | +0.013838 ±0.006308 | 88.14% |
| ≥25.5 | 3.15% | +0.006480 ±0.001008 | 94.86% |

S/N is SExtractor FLUX_AUTO/FLUXERR_AUTO, not ngmix shape S/N. The bright
small galaxies do not show the positive residual. In the main modest-crowding
bin, intrinsic magnitudes 24–25.5 contribute +0.010566 ±0.002657, but
23–24 also contribute +0.006638 ±0.002335. Therefore this is not confined
to the faintest or lowest-S/N tail.

The intrinsic ≥25.5 group passes measured magnitude <25 only after a large
flux boost; its mean measured/truth flux ratio is 4.72. Blend contamination,
noise, detection association and selection can contribute to such boosts.
This catalogue-only test does not assign them individually.

## Direct neighbour effect on these objects

The existing matched four-state images hold each target's own shear and image
noise seed fixed while reversing the opposite group's shear. On the common
usable cohort, small galaxies cross the radius threshold in **4090 edge
comparisons**, or **0.488% ±0.013 percentage points** of the union-selected
comparisons that pass magnitude <25 in both states. These are edge comparisons,
not independent unique galaxies. Average absolute radius movement over all
common-usable, both-magnitude-passing small objects is 0.0346 pixels;
7.98% move by >0.1 pixels. Both increase with crowding.

The exact neighbour-induced selection-response terms, including the magnitude
cut and ratio normalization, are +0.003234 ±0.000928 for target group A and
+0.003402 ±0.000534 for B, normalized within the intrinsic-small group and
target group. Thus changing neighbour shear really changes selection without
changing target shear or image-noise seed. This is direct evidence for a
neighbour-related selection effect, not just a crowding correlation.

In the original global-error normalization, the intrinsic-small population
has own-group error +0.015236 ±0.005614 and opposite-group error
+0.006291 ±0.002571; these sum to +0.021526. Own-group includes same-group
neighbours and is not a pure flow error. Opposite-group must not be doubled
to estimate all-neighbour error. The selected-shape model's opposite-group
selection is fixed, so the measured neighbour-induced selection term is
missing from that model. See the
[matched experiment](../constant_split_20261105/findings.md#completed-matched-shear-response-decomposition).

## Why a size cut changes the shape response

The constant scenes reproduce the sharp shape–size correlation previously
seen in the standalone flow investigation. Define descriptive alignment
as sum(e_measured dot e_intrinsic)/sum(|e_intrinsic|²), pooling the original
plus and minus shear legs. This is not a zero-shear statistic or a shear
response; it is used only to show the conditional shape distribution.

For Re 0.15–0.25 arcsec and crowding <0.1, the alignment slope is:

| Measured radius, pixels | Alignment slope |
|---|---:|
| 2.5–2.6 | −0.519 ±0.008 |
| 2.7–2.8 | −0.367 ±0.011 |
| 2.8–2.9 | −0.119 ±0.004 |
| 2.9–3.0 | +0.159 ±0.009 |
| 3.0–3.2 | +0.513 ±0.003 |
| 3.2–3.5 | +0.715 ±0.003 |

A negative dot product of spin-2 ellipticities means a tendency toward
perpendicular physical axes. The size cut therefore selects a very different
distribution of shape errors: the correlation changes rapidly around the cut.
The prior standalone comparison found that the flow smooths this relation too
strongly while reproducing mean radius transport. This supports a conditional
shape–size modelling failure, rather than simply an incorrect mean size.
See the preserved [standalone investigation](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-04--why-the-flows-response-fails-around-3-px-it-under-models-how-measured-shape-and-measured-size-share-noise-for-psf-sized-galaxies).

The existing optimizer audit separately found signed ngmix T solutions that
can reverse fitted shapes on identical pixels. That is relevant to the
conditional shape distribution, but no experiment here establishes that
negative T causes the mean response error. It cannot explain FLUX_RADIUS
variation directly: that radius comes from SExtractor, before ngmix fitting.

![Crowding localization and conditional shape alignment](mechanism.png)

## Interpretation and discriminating follow-up

The supported account is: PSF broadening puts small galaxies near the size
threshold; selecting measured size selects correlated shape errors; the flow
under-models that conditional structure in the standalone experiment; and
neighbour shear additionally changes measured selection on the constant
scenes, a response absent from the current generated-selection model.
Neither noise alone nor severe blending alone is established as the cause.

To quantify **why individual radii are large**, the discriminating next image
experiment is a matched two-by-two intervention: target alone / full
neighbour scene, each with noise absent / original noise. Use identical PSF,
source profiles, pixel phase, photometry configuration and matching rules;
first replay the original noisy full-scene measurements. Sample fixed truth
strata, including unselected controls; retain non-detections as outcomes.
Then repeat noise and seeded shape fits separately to distinguish image noise
from optimizer scatter. New coherent-shear legs are needed to test whether
each intervention removes the response discrepancy, rather than only the
radius enlargement. These interventions were not performed in this follow-up.

## Validation and artifacts

- CPU jobs 17754665 and 17755490 completed 0:0 in 30 and 23 seconds.
  The second adds the magnitude split and explicit empty-bootstrap reporting.
- Every original coherent simulation/model selected count and shape sum
  replays; global own/opposite errors and bootstrap SEs agree within 2e-12.
- Disjoint truth-cell contributions add exactly. Common-U shape plus
  selection terms reproduce the finite-difference selected means.
- 10,000 paired whole-case bootstrap resamples, seed 20260925; fixed model
  draws. Four cases and exploratory subgroup selection limit precision and
  generalization. No independent acceptance claim.
- Crossmatch and shape IDs are unique and joined one-to-one. All 16 catalogues
  have zero crossmatched detections missing shapes; 1166–1624 shape rows per
  catalogue lack crossmatches and are excluded. The roughly 591k–593k parent
  rows without a crossmatch per catalogue are unusable, not silently assigned
  valid measurements. Counts appear in report receipts.
- The analytic PSF summary uses exact noise.csv parameters and the original
  renderer's truncation rule. Sparse alignment bins with undefined bootstrap
  ratios report null SE and the number of undefined replicates.
- Source: [analysis](analyse.py), [numerical report](report.json),
  [case sums](case_sums.npz), [figure PDF](mechanism.pdf), [scope](rule.md#frozen-scope).
  No models, production measurements or scientific cuts changed.

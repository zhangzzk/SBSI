# Measured diagonal-gap localization

## Population and estimator

Use the same5,412,739 IDs passing the central measured cut in all four exposures
across40 cases. Finite-difference each measured shape over±0.02, then average
on this common population. The full matrix exactly reproduces the prior
[same-ID result](../measured_response_symmetry_20261107/findings.md#same-galaxy-measurement-check).
Errors are paired40-scene delete-one jackknife standard errors. Groups are
exploratory, correlated, and sometimes defined using the response itself;
none is a proposed acceptance cut or a causal error allocation.

## Intrinsic orientations do not predict the gap

Apply exact reduced-ellipticity shear composition to the intrinsic shapes of
the same selected IDs. Their ideal R11−R22 is−0.0000799±0.0001145, compared with
the measured+0.0086336±0.0031973. This rules out a simple imbalance in the
intrinsic spin-four moment as an explanation in the ideal-response calculation.
It does not model blending, pixel integration, detection, or noisy fitting.

## Large response excursions

Define excursion as the maximum absolute entry in each object's measured
2×2 response. Contributions below sum to the total diagonal gap; each uses
the original common-population denominator.

| Maximum absolute response entry | Population fraction | Contribution to total R11−R22 |
|---|---:|---:|
| <2 | 71.53% | +0.002131 ±0.000203 |
| 2–5 | 11.46% | +0.000669 ±0.000308 |
| 5–10 | 3.37% | +0.000258 ±0.000474 |
| 10–20 | 6.68% | +0.001951 ±0.001247 |
| ≥20 | 6.95% | +0.003625 ±0.002786 |

The large excursions account for much of the uncertainty and point estimate,
but the gap is not exclusively a handful of extreme objects. Conditioning on
response changes the population; do not use this table to justify trimming or
claim separate physical bias components. The controlled image replay checks
whether optimizer behavior contributes to those changes.

## Size, brightness and subpixel checks

No single brightness bin accounts for the entire difference. Galaxies with
mean measured radius3–4pixels are64.85% of the population and contribute
+0.006250±0.003456 to the total gap. Radius≥7pixels is2.23% and contributes
+0.001051±0.000298. These are exploratory correlated subdivisions, not evidence
of a unique size-dependent mechanism. All intrinsic-ellipticity groups have
positive point estimates with broad errors. Subpixel phase groups do not yet
establish a coherent coordinate-dependent pattern. Complete numbers are in
`report.json`; group assignments use four-exposure mean photometry, median
positions, intrinsic ellipticity, or measured response as explicitly labelled.

## Validation

CPU17821702 completes catalogue reduction. All joins enforce unique IDs;
unmatched rows are counted in160 receipts. Sum data are retained at the
project cache `measured_gap_localization_20261107/sums.npz`. The pooled matrix
replays to1e-12. No images, fits or model parameters changed in this analysis.

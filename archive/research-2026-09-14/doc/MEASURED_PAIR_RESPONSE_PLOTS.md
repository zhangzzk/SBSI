# Measured per-pair radius and flux responses

## Definition and population

These plots contain simulation measurements only, with no model predictions,
fitted `b*e` targets, or empirical corrections. Each signed response is the
forward difference from zero shear to the pair's actual applied shear norm
(approximately 0.05), along its applied direction. These are finite secants,
not independently measured infinitesimal derivatives.

The existing shared-Re>0.5 validation sample contains **1,120,160** matched
galaxies in **40** held-out cases, detected and usable in both legs, with no
additional measured selection cuts. Original catalogue matching reported
46,189 unmatched zero-leg rows, 43,131 unmatched sheared-leg rows,
5,117,032 domain/detection/nonfinite exclusions, and one zero-shear exclusion.
All reference keys match the exact-radius cache and the newly loaded zero-leg
flux catalogue. All four plotted responses are finite on every retained pair.
This population is conditional on successful measurements in both legs.

Radius uses the original measured flux radii, converted with 0.2 arcsec/pixel.
The sheared cache has measured MAG_AUTO but no raw FLUX_AUTO column. Therefore
F1/F0 is reconstructed as `10**(-0.4*(mag1-mag0))`, assuming the common magnitude
calibration. F0 is the original zero-leg FLUX_AUTO. Its MAG_AUTO + 2.5 log10(F0)
ranges from 29.99994865 to 30.00005126, consistent with zero point 30 and saved
column precision. Reconstructed sheared flux inherits the magnitude precision.

## Figures and interpretation

- [Absolute radius response](../figures/measured_pair_response_v1/radius_response.png): Delta r / Delta g, arcsec per unit shear.
- [Fractional flux response](../figures/measured_pair_response_v1/fractional_flux_response.png): (F1-F0)/(F0 Delta g).
- [Absolute flux response](../figures/measured_pair_response_v1/flux_response.png): (F1-F0)/Delta g, original catalogue flux units.
- [Log-radius response](../figures/measured_pair_response_v1/log_radius_response.png): Delta ln r / Delta g, matching the earlier response coordinate.

Each also has a PDF with the same stem. Nine properties are shown: intrinsic
ellipticity along shear, intrinsic ellipticity magnitude, measured ellipticity
magnitude, true effective radius, measured radius, R_blend, true magnitude,
measured magnitude, and Sersic index. Measured properties are from the zero leg.

Blue points give means and 1-SE error bars from 2,000 whole-case bootstrap
resamples (seed 20260906). Orange gives medians; the blue band is the 16–84%
spread of individual pairs, and dotted lines give 5–95%. Statistics use all
finite values in 12 marginal quantile bins. Y axes show the central 1–99%
range with padding and X axes the central 0.5–99.5%; extremes remain in the
statistics and saved data. Bands are not uncertainty bands for the mean.

The mean radius response changes from -0.28559 +/- 0.00554 to
+0.31988 +/- 0.00339 arcsec per unit shear between the bins centered at
intrinsic e_parallel approximately -0.5 and +0.5. Fractional flux response has
substantially broader scatter at faint magnitudes. These are marginal trends,
not controlled partial dependences. Trends versus noisy zero-leg measurements
can include regression-to-the-mean effects, not only physical shear response.

## Directional response split by properties

The follow-up keeps directional ellipticity on the X axis and splits each of
six other properties into quartile intervals (duplicate edges merged; ties
can make populations unequal). It uses the same 1,120,160 rows, 12 common
ellipticity bins, all finite response values, and 2,000 whole-case bootstrap
resamples. Each curve is a measured mean with 1-SE bars, not a fitted model.

- [Radius response versus directional ellipticity](../figures/measured_pair_response_v1/radius_response_directional.png)
- [Log-radius response versus directional ellipticity](../figures/measured_pair_response_v1/log_radius_response_directional.png)

PDFs share the same stems. Descriptive OLS slopes with a fitted intercept are
saved separately; no fit is applied to model predictions or targets.

| Property, lowest to highest quartile | Radius-response slope, low / high | Log-radius-response slope, low / high |
|---|---:|---:|
| True effective radius | 0.4479 / 0.7323 | 0.5142 / 0.6075 |
| True magnitude (bright to faint) | 0.6769 / 0.3995 | 0.6534 / 0.4125 |
| R_blend | 0.6689 / 0.5394 | 0.6561 / 0.4772 |
| Measured zero-leg radius | 0.3643 / 0.8578 | 0.4728 / 0.6366 |
| Measured zero-leg magnitude | 0.7380 / 0.3782 | 0.6577 / 0.4262 |
| Sersic index | 0.6644 / 0.4660 | 0.6535 / 0.4411 |

Radius-response slopes are in arcsec per unit shear per unit ellipticity;
log-radius slopes are per unit shear per unit ellipticity. For log-radius,
highest-minus-lowest slopes are +0.09333 +/- 0.00862 (true size),
-0.24089 +/- 0.00683 (true magnitude), and -0.17896 +/- 0.00766 (R_blend).
These SEs use paired whole-case bootstrap contrasts. Thus the earlier flat
marginal means concealed property dependence of the directional response.
Dependence on true size weakens, but persists, in log-radius units.

These are stratified associations, not independent causal effects: other
properties are not held fixed or reweighted. Conditioning on measured
zero-leg quantities can introduce regression-to-the-mean effects. The
approximately linear curves do not prove the b*e approximation is adequate
for the full conditional distribution or for selection response.

External files: `directional_profiles.py`, `directional_profiles.csv`,
`directional_slopes.csv`, and `directional_summary.json` in the same artifact
root below. Slurm **16326945 completed in 22 s**, exit 0, empty stderr.
Assertions verify sample size, 40 cases, finite inputs, populated cells, and
finite bootstrap slopes. Both figures were visually inspected.

## Reproducibility and validation

External artifact root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_audit_v1/measured_property_plots_v1/`.
It contains `plot.py`, `measured_pairs.parquet`, `profiles.csv`, `summary.json`,
and Slurm logs. Input paired properties come from the sibling
`results/per_galaxy.parquet`; exact radii and shear norms come from
`exact_radius_reference.parquet`. No trained model is loaded.

Slurm **16326906** completed in **34 s**, exit 0, empty stderr. Assertions
verify exact key coverage, one-to-one joins, shear norms, and reproduction of
the previously saved measured log-radius response. Both main PNGs were visually
inspected. Initial job **16326895** failed because the sheared cache lacks raw
flux; the final script explicitly records the magnitude-based reconstruction.
No production code or training setting changed.

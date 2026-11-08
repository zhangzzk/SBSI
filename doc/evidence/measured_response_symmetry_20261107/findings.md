# Measured response symmetry audit

## Conclusion

The stronger measured signal is unequal diagonal response, not a resolved
violation of transpose symmetry. At radius>3 pixels (0.60 arcsec), magnitude<25,
the paired40-scene result is R11−R22=+0.009653±0.003571 (2.70 standard errors),
whereas R12−R21=−0.002953±0.003622 (0.82 standard errors). Errors below are
whole-scene delete-one jackknife standard errors; bootstrap intervals use10000
paired resamples. These are exploratory reused-data checks, not independent
acceptance or evidence that an exact symmetry should be imposed.

## Matrices and model architecture

Rows are measured shape components and columns are applied shear components.
Use separately normalized±0.02 legs and pool scene sums before forming ratios,
as in [response conventions](../../CONVENTIONS.md#6-response-estimators).

| Response | R11 | R12 | R21 | R22 |
|---|---:|---:|---:|---:|
| Measured, original selection | 0.907138 | −0.003516 | −0.000563 | 0.897485 |
| Current weighted model, generated inputs and predicted usability | 0.899866 | −0.002308 | 0.000755 | 0.900488 |
| Current model, measured photometry and usability | 0.906078 | −0.002003 | 0.000969 | 0.906143 |
| Measured, same selected IDs in all four exposures | 0.898726 | −0.004231 | 0.000485 | 0.890093 |

The full model is not symmetric by construction: its measured-input
R12−R21=−0.002972 is almost the same as the measured−0.002953. The Cartesian
conditional flow has freely learned two-component couplings. Only the
fixed-shape, zero-velocity derivative in
`sbsi/disk_response_transport.py::infinitesimal_response_matrix` is necessarily
transpose-symmetric; its diagonal entries need not be equal. That primitive's
symmetry does not constrain the full selected response. Near-equal model
diagonals are an observed prediction, not an enforced identity.

## Same-galaxy measurement check

Re-read all160 measured catalogues and reproduce all original sums, then retain
IDs selected and usable in every one of the four axis/sign exposures.

| Measured population | R11−R22 | R12−R21 |
|---|---:|---:|
| Each exposure's own selection | +0.009653 ±0.003571 | −0.002953 ±0.003622 |
| Same IDs in all four exposures | +0.008634 ±0.003197 | −0.004716 ±0.003284 |
| Difference of those estimators, paired | +0.001019 ±0.001222 | +0.001762 ±0.001070 |
| Same IDs within each axis's plus/minus pair | +0.009503 ±0.003154 | −0.004602 ±0.003223 |

The common-four sample contains5,412,739 galaxies,97.04% of each original
selected exposure. Its diagonal gap remains2.70 standard errors from zero;
95% bootstrap interval[0.002535,0.014739]. Its point estimate retains89.4% of
the original gap. The paired change is unresolved, so membership changes do
not account for most of the observed gap in this check. Conditioning on all
four outcomes changes the population; the difference is an accounting
remainder, not a causal allocation of selection bias. The standard estimator
remains unchanged. Common-four diagonal differences are positive across all
nine target cuts,0.00625–0.00865, but the cuts are highly correlated.

## Stability and pipeline audit

The original four scenes have diagonal difference−0.003286±0.003103; the
additional36 have+0.011090±0.003886. Across all40,27 individual differences are
positive, and deleting any single scene leaves the pooled difference positive,
0.008357–0.011156. This is not driven by one scene, but the original small
pilot did not establish this effect.

All80 paired configuration comparisons pass, with only expected paths changed.
All160 PSF pixel arrays are identical; both input PSF ellipticity components
are zero. Noise catalogues are byte-identical,80 prior truth-axis verification
receipts pass, and original rendering/measurement source hashes still match.
These checks rule out those cross-axis configuration differences, not all
possible image-generation or measurement errors.

The fitter uses an unseeded `np.random.RandomState()`. Matching rendering seeds
does not pair optimizer initialization. There are no rotated-scene pairs in
this run. Circular PSF and square pixels alone do not enforce equal shear-axis
responses: interchanging g1 and g2 involves a45-degree physical rotation.
Neither observation yet identifies a cause of the diagonal gap.

The PSF light centroid is(24,24) in a48-square stamp while the declared Jacobian
origin is(23.5,23.5). Inspection of installed ngmix `gmix_convolve_fill` shows
that convolution subtracts the fitted PSF centroid. This coordinate-origin
difference therefore does not itself impose a galaxy centroid shift; it is
not established as a bug. Source hashes and moments are in `psf_moments.json`.

## Validation and next discriminating check

CPU17821487 completes0:0 in14s; packed CPU17821503 completes0:0 in5m03s.
`verification.json` tests axis orientation and both contrasts using a known
nonsymmetric matrix and unequal leg masses. Every original measured sum
replays at rtol1e-12/atol1e-8; common-four counts are identical in all legs,
and full=common+remainder closes in every resample. Full matching receipts,
input hashes, scene bootstrap intervals and all14 cut results are retained in
`report.json` and `membership_report.json`. Across the160 catalogue pairs,
228,683 shape rows without crossmatches are excluded; zero crossmatch rows
lack shapes. Parent rows without crossmatches are not usable.

The next discriminating measurement experiment is a small controlled replay:
first quantify repeat-fit variation on unchanged images with fixed, recorded
optimizer seeds; then compare original scenes with45-degree rotated scene
counterparts under matched noise treatment. Use identical IDs and the same
selection conventions and inspect component transformation explicitly.
This would separate fitting variability from orientation-dependent rendering,
pixelization, blending and measurement effects. It has not been run here, and
no single cause has yet been localized. Do not fit away or symmetrize the gap.
No model updates, image rerenders or fitter calls in this audit. Broad17788286
remains held and automated follow-ups remain disabled.

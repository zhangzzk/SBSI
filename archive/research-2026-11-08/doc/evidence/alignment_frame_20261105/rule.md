# Controlled frame experiment and identical-galaxy measurement test

## Purpose and fixed scope

Owner requests both an explanation of FS2/GalSBI differences and attempts to
improve GalSBI. Do not equate differing populations with differing observing
conditions. Two GPU workers total; no other SBSI jobs active at submission.
Preserve all retained presets. New weights are experimental artifacts only.

## Cartesian versus intrinsic-frame density

Fresh matched GalSBI fits: same40 randomized training cases used in the preceding
diagnostic, all valid measurements of both legs, same40 validation cases.
Same full four-output physical-disk affine architecture, all eight physical
inputs retained, same target/context reference standardization and all radius/
flux transforms. Use common isotropic disk-coordinate scale and zero coordinate
mean for both initializations. Same seed202611053111, batches, training rows,
AdamW, clipping5,120 epochs of two million rows, initial rate3e-5 and identical
ReduceLROnPlateau(factor0.5, patience8, absolute threshold1e-4, min_lr1e-7).
Select each checkpoint solely by NLL on a fixed500k validation sample, seed
202611052701. No measured training cut, input-range restriction, guard or
alignment penalty. The comparison is developmental, not untouched acceptance.

Cartesian arm models world-frame shapes. Frame arm rotates measured shape
into the direction of the leg's intrinsic shape; inverse rotation returns
draws to world coordinates before any response or cut. Keep original e1/e2
inputs so angular image anisotropy remains representable. Rotation determinant
is1; standardized/world likelihoods share the same density reference. For
intrinsic norm<1e-8 use identity rotation and count such rows. No division by
ellipticity magnitude in the target values and no shape clipping.

Before launch, test rotation/inverse, norm and Jacobian preservation, likelihood
equivalence to an explicitly rotated target, and saved/reloaded predictions.
Artifacts have a custom experimental format so a standard loader cannot
silently interpret local-frame weights as a world-frame model.

Evaluate selected endpoints on all40 validation cases with two banks of16
antithetic draws and common random numbers between shear legs. Save the same
seven hard-cut response moments and shape-size profiles as the transfer audit.
Use10,000 whole-case bootstrap resamples; report absolute response error,
alignment, bin counts and NLL. Minimum evidence for an improvement: better
validation alignment without a material density loss or degradation hidden
at neighbouring radius cuts. This is not joint constant-shear acceptance.

## Identical-galaxy observing-condition experiment

Render isolated galaxies with identical intrinsic parameters and flux under
the two analytic PSF prescriptions, then cross with zero, r-like and i-like
noise. Use common noise fields and fitter seeds between interventions; same
centred65 measurement code. Fixed grids: circularized Re[0.15,0.20,0.25],
Sérsic n[1,2,4], intrinsic reduced ellipticity amplitudes[0.1,0.3,0.5],
sky angles[0,45]degrees, magnitude[23,24,25,25.8], four noise/fitter seeds.
Record all finite/failure/negative-T outcomes, with no T cut. This experiment
does not reproduce SExtractor detection or FLUX_RADIUS selection; it tests
whether observing conditions alone can change fitted shape branches on the
same galaxies. Keep this distinction explicit in reporting.

## Follow-up Gaussian-PSF control, declared before launch

After the Moffat intervention found a PSF-only alignment change even without
added noise, repeat the identical216-context/same-seed grid with Gaussian PSFs.
Set their intrinsic T=2sigma² to the previously audited production one-Gaussian
PSF-fit T values0.4937487425 and0.4177950357arcsec². Keep galaxy rendering,
pixel integration, stamps, fitter, noise and all other settings unchanged.
This tests a correctly specified PSF family, not an alternative calibration
pipeline or a matched-FWHM observation. Report negative-T and alignment by
noise level without cutting failed/negative fits from the raw record.

## Follow-up three-component PSF fitter, declared before launch

The Gaussian-rendering control removed all negative-T fits without added noise.
Now retain exactly the original Moffat images/noise/context grid and replace
only the diagnostic PSF guesser's ngauss=1 with ngauss=3. Keep its EM fitter,
ntry2, galaxy Gaussian model/prior and all seeds unchanged. Raw failures remain
recorded. This tests whether improving the PSF approximation removes the
pathology while leaving the rendered observation fixed. It does not change
production measurement or propose a calibrated replacement without retraining
and constant-shear validation.

# Radius dependence of the frozen Cartesian flow

## Question and fixed settings

The completed 40-scene tests show small main-cut bias but approximately +2.4%
at measured radius >=0.75 arcsec in both shear directions. All three training
seeds show the tighter-cut residual. Map this dependence and its intrinsic-size
contributions without selecting a new checkpoint or fitting a correction.

Keep Cartesian seed611, original C8 and physical-moment emulator, all40 scenes,
per-leg normalization,64 antithetic draws,1024-row blocks and original CRN seeds.
Reuse the completed g1 and g2 images. Evaluate magnitude<25 and<25.8, radius
thresholds0.45..0.90 arcsec at0.05 intervals, plus magnitude-only and uncut.
Use >= except the existing >0.60 cut. Bin by intrinsic circularized radius
at0.15,0.25,0.4,0.7 arcsec without imposing a truth-property selection.
Report predicted and actual usability with generated output selection.
Measure each intrinsic bin's contribution using the full selected denominator;
do not confuse this with a bin-normalized response or standalone flow error.

## Validation and resources

For every case/axis, require replay of all six overlapping old selections after
summing over intrinsic bins (simulation atol1e-8,rtol1e-12; stochastic model
atol2e-7,rtol2e-10). These allow summation-order roundoff, not empirical offsets.
Preserve source, checkpoint and catalogue hashes and unmatched-row receipts.
One case on each axis gates the remaining78 tasks. At most six GPUs total;
all previous study GPU arrays have completed. No new image simulation required.
Whole-case paired10000 bootstrap samples condition on fixed models and measured
catalogues. These are development diagnostics, not independent acceptance.

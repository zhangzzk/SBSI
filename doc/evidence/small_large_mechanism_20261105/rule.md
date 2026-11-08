# Small intrinsic size and large measured size: mechanism follow-up

## Frozen scope

Use GalSBI-dev, cases 1000/1040/1080/1120, the original four-state
constant-shear measurements and saved per-object model moments. No retraining,
correction, changed usability, or new truth cut in the scientific estimator.
The main selection is FLUX_RADIUS > 3 pixels and MAG_AUTO < 25. Intrinsic
size is circularized Re; pixel scale is 0.2 arcsec. Report simulation minus
model absolute R11 contributions with separate selected denominators.

## Questions and checks

1. Replay the original response and localize own/opposite-group error across
   disjoint intrinsic size and crowding bins, including a stricter low-crowding
   threshold 0.01. Fixed truth populations must sum to the global response.
2. Describe measured sizes, photometric signal/noise, measured/truth flux,
   flags and match distances. These are associations, not noise/blend labels.
3. Measure size-cut crossings when only the opposite group is sheared,
   holding target shear and image-noise seed fixed. Report shape and selection
   response separately on common-all-four usable objects. Do not double
   opposite-group terms into a full-neighbour estimate.
4. Inspect alignment of measured with intrinsic ellipticity versus measured
   size, separately for low/high crowding. This is a conditional descriptive
   statistic, not an alternative response estimator.
5. Account for the actual Moffat PSF before describing a measured radius as
   inflated by noise or blending. Intrinsic Re and SExtractor FLUX_RADIUS are
   different measurements. An analytic PSF radius is not an object-specific
   noiseless SExtractor prediction.

## Uncertainty and limitations

Use 10,000 paired whole-case bootstrap resamples, seed 20260925. Four scenes,
fixed model draws, and prior development use limit interpretation. Original
optimizer randomness remains in the measurements. Catalogue associations
cannot identify a causal fraction due to noise versus neighbour light;
that needs matched image interventions, preserving the photometry pipeline.
All catalogue scans run on Slurm CPU allocations.

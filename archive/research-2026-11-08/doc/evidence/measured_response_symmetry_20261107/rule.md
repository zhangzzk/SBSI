# Measured-response symmetry investigation

## Owner request and definitions

On2026-11-07 the owner asks to investigate apparent measured-response asymmetry
before further model work. Distinguish transpose symmetry, R12=R21, from equal
diagonal response, R11=R22. Neither should be silently imposed on a noisy
finite-scene selected measurement. No model training or adjustment is authorized
by this diagnostic, and none is performed.

Use the same40 scene identities, both axes±0.02, all nine target cuts and five
ancillary cuts. Report original4/additional36/all40 separately. Pool numerator
and mass before ratios; retain each leg's selected mass. Reuse whole-case
resampling weights across axes/cuts with10000 bootstrap draws, seed202611071940,
and delete-one-case jackknife. Report matrices, diagonal/offdiagonal contrasts,
bootstrap intervals, per-scene signs and delete-one-scene sensitivity. These
are exploratory stability checks on reused data, not nine independent tests.

## Pipeline and architecture checks

Compare all80 paired axis configurations, allowing only the original path
substitutions. Read all160 PSF images and compare pixel arrays. Verify identical
noise catalogue bytes, circular input PSF parameters and all80 existing g2 truth
verification receipts. Check recorded rendering/measurement source hashes.
Record whether ngmix initialization is seeded; matched renderer seeds do not
by themselves pair optimizer randomness or prove all pixel noise realizations
identical.

Inspect the Cartesian flow and the transport primitive separately. The zero-
velocity fixed-shape Mobius derivative is symmetric; this does not impose
transpose symmetry or equal diagonals on the full selected flow+transport
response under separately perturbed scenes.

## Measured membership check

Read existing catalogues without rerendering or refitting. Reproduce every
original measured sum before computing responses on (a) each leg's own selected
sample, (b) IDs selected/usable in all four axis/sign exposures, and (c) IDs
selected/usable in the two legs of each individual axis. Define the remainder
as full response minus common-all-four response, so that the exact identity
holds for every resample. This remainder combines changes of membership and
normalization relative to a changed reference population; it is not a unique
physical selection-error component. Report selected counts/fractions and all
matching exclusions. Common-ID conditioning must not replace the main estimator.

One short CPU job audits sums/configurations/PSFs; one packed CPU job reads all40
cases and performs membership reduction. No GPUs, new images or fitter calls.
Broad17788286 remains held and automated follow-ups remain disabled.

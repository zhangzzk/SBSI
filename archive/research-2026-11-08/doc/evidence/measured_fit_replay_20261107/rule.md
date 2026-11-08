# Controlled measurement replay

## Question and fixed sample

The owner asks why the measured diagonal responses differ. Test fitting
randomness and convergence on unchanged images before rerendering or changing
model training. Use cases1000,1001,1041,1081 and512 uniformly sampled parent IDs
per case, selected with seed20261107+case from the common central-cut population
usable in all four original exposures. This is a diagnostic subset, not a new
acceptance sample. Keep original membership and measured photometry fixed.

## Interventions

Repeat the original65-pixel centred cutout, detected position, PSF, noise
estimate and ngmix Gaussian fitter. Inject recorded per-object RNG seed A,
repeat with seed B, then reuse seed A with ftol/xtol reduced from1e-5 to1e-10.
Keep maxfev4000. Seeds are paired across axes and signs. The original fitter
source is cloned with exactly three asserted substitutions: signature, RNG
seed injection and optional LM parameters. Production source is untouched.
Record returned g, status flags, evaluation count and exceptions. Repeat the
first fit exactly as a determinism gate. Complete first case before permitting
remaining three packed workers; require98% finite interior fitted shapes.

Compare responses on common valid refits and quantify exclusions separately.
Report changes paired by galaxy and scene; four scenes cannot establish a
population-level null or independent acceptance. Sample response itself is
noisy: primary purpose is a paired change from the historical fit, and changes
between controlled settings. A few flagged/outlying fits are not permission to
alter catalogue cuts to force agreement.

## Execution and provenance

Measurement implementation lives in BlendEMU at
`/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts/diagnose_constant_fit_replay.py`.
Outputs go to the project cache of this study's name; documentation and analysis
live in SBSI. Four CPUs,32GB,30-minute limit, no GPUs. Job17821683.
Large images are read through FITS memory mapping; each case keeps only its
small output. Preserve original catalogues and all trained models. Broad
array17788286 stays held; automated checks stay disabled.

## Isolated orientation control

Job17821716 uses two CPUs to fit noiseless isolated Sersic galaxies with
16 uniformly spaced angles and four symmetric subpixel phases per ring.
Cross n=1/4, circularized radius0.35/0.7arcsec and original truncated Moffat/
Gaussian PSF controls, fixed intrinsic |g|=0.5 and magnitude24. All four shear
legs use tight fits and paired optimizer seeds. Use the production sky weight
explicitly despite no added noise. This separates basic pixel/fitter anisotropy
from noise, detection and neighbour interactions. It is not the production
population and cannot by itself rule out noisy blended measurement errors.

## Adaptive optimizer continuation diagnostic

After the first scene reveals seed sensitivity, sample32 seed-sensitive and32
stable-control first-leg objects. Compare default A/B, tight A/B, and continuation
from each default solution with its exact fitted PSF and unchanged prior.
Evaluate the objective and shape movement; this enriched sample is not a
population frequency estimate. The continuation tests premature convergence
without rerendering or changing image weights.

Initial runs17821736/17821743 on th-cl-cappm18 fail a replay check against the
th-cl-dell07 pilot: first a1.74e-7 discrepancy, then a0.1273 discrepancy even
with a1e-4 tolerance. Do not call this cross-node replay validated. A requested
same-node job17821749 was cancelled while pending. The revised diagnostic
records all cross-node differences and instead requires every live default
fit to repeat exactly within its own allocation, before comparing its own
continuation. This changes the control to a within-job comparison; it does
not relax the claim of cross-node reproducibility. Keep original sample IDs
and report the discrepancy as an additional reproducibility limitation.

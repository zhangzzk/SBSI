# Why the GalSBI alignment relation is underfit

## Fixed diagnostic before execution

Compare frozen NLL-only and adopted FS2/GalSBI flows, without training or
constant-shear fitting. Use all 40 harmonized validation cases and the first
40 cases in the recorded randomized training-case order. Restrict intrinsic
circularized Re to [0.15,0.25) arcsec. Use the same matched usable g0/g05 keys
as the preceding transfer audit, but only g0 measurements/contexts here.
Report dropped pairs and selected counts. This is not a shear response test.

Each model uses two independent banks of 16 antithetic draws, shared across
checkpoints for a given case/block/bank. Each branch bins its own measured
radius and applies its own magnitude<25/<25.8 cut. Record alignment,
negative/positive dot-product contributions, counts and perpendicular shape
projection. Fixed 0.1-pixel radius bins from2 to5 with open outer bins.

Intrinsic strata: Sérsic n edges1,2,4; true magnitude23,24,25;
crowding nbr_close_100 edges0.01,0.1,1; four equal intrinsic ellipticity-angle
sectors; ellipticity magnitude edges0.2,0.5. Also Sérsic×magnitude cells.
These are diagnostic partitions, not selection changes to production.
Compare train/validation and NLL/adopted within each family, then assess
whether the population-averaged error hides one subpopulation or is broad.

Evaluate observed-target log density under each model on these same rows.
Record the density change when rotating both intrinsic and measured shape
vectors by90 degrees in ellipticity space (45 degrees on the sky), leaving
scalar features fixed. This tests learned angular dependence; pixelized image
measurements are not guaranteed exactly rotation invariant, so it alone does
not establish an architectural defect.

Use10,000 whole-case bootstrap resamples for key contrasts, fixed models.
No claim that case errors include shared FS2 noise, fitting randomness or
training uncertainty. Verify additive partitions and replay existing measured
validation alignment. Two GPU workers maximum; no other SBSI GPU jobs at
submission. No image changes, retraining, or empirical correction in this stage.

## Analysis clarification before completed profiles

Decompose each global alignment difference over the16 Sérsic×true-magnitude
cells. With q the fraction of intrinsic ellipticity power in each cell,
use the exact symmetric identity Delta A = sum(mean(q)*Delta A_cell) +
sum(mean(A_cell)*Delta q). Report unsupported power explicitly and condition
the decomposition on shared support. This separates coarse compositional
changes from within-cell shape error; it is not a causal decomposition of m.

## Bounded learning intervention, declared after frozen-profile results

Frozen results show the GalSBI mismatch already on training data, with the
NLL-only model worse than the adopted model. Start two diagnostic continuations
from the same selected GalSBI NLL-only weights and saved AdamW moments. Retain
architecture, feature/target transformations, batch8192 and clipping5. Both
arms use the same40 training cases (all independently usable rows of both
legs, not the matched-pair diagnostic subset),20 epochs, four million rows
per epoch, and fixed reset learning rate1e-4. No guard or alignment penalty.

Control: sample the entire input population. Targeted: sample only intrinsic
circularized Re[0.15,0.25). No measured-property selection or cut enters
training. The input restriction changes training emphasis without changing
the conditional target density within that input range. Random seeds are
shared across arms; both have the same number of gradient updates.

Evaluate fixed endpoints, not a selected best-alignment checkpoint. Measure
validation NLL on a fixed500k sample of all40 validation cases before/after,
and the alignment profiles on all40 validation cases with the preceding
probe's exact common draw streams. Save two16-draw banks and case moments.
The difference between arms tests input-population emphasis under this reset
schedule. Improvement over the starting model also includes additional updates,
the reset rate and use of a40-case training subset; it does not by itself
identify which caused the original training failure. A negative result cannot
prove insufficient model capacity. These are disposable diagnostic models,
not candidate replacements or a constant-shear acceptance experiment.

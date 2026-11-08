# Original joint residual localization, 2026-11-05

Owner requested further localization after measured-mask substitutions did not
explain the original error. No training, model changes, or empirical correction.

## Neighbour-induced selection response

Use the existing 40 emulator-development cases (random-direction neighbour
shear h=0.2), unchanged original matched usable labels, bright-neighbour
exclusions, geometry and nearest-20 secondary pool. Join actual radius/magnitude
in both legs. Compute the projected selected influence response, including
selection migration, instead of restricting the shape difference to the
baseline-selected sample. Decompose exactly into baseline-selected shape
response plus selection migration minus the selected-mean normalization term.
The emulator's baseline shape prediction is unchanged. Replay the old target-cut
pair counts, labels and model sums before interpretation. Use uncut, magnitude,
radius and combined cuts; bin by intrinsic circularized radius as a diagnostic.

This test preserves measured shape/selection coupling on the simulation side.
It tests a term omitted by the existing shape-only standalone emulator check.
It is still a forward h=0.2 random-direction half-neighbour experiment, with
common usability. It cannot be pasted into the h=0.02 coherent-shear budget,
and does not measure usability migration or prove small-shear transport.

## Joint residual by fixed truth population

Replay original true-U/generated-selection constant-shear model on cases
1000,1040,1080,1120, with its original 64 draws and block seeds. Keep generated
shape, radius/flux, selection and emulator inputs together. Accumulate per-leg
counts, shape and emulator shift by fixed truth circularized size
(<0.15,0.15-0.25,0.25-0.4,0.4-0.7,>=0.7 arcsec), true magnitude
(<24,24-25.5,25.5-27.5,>=27.5), and nbr_close_100
(<0.1,0.1-0.3,0.3-1,1-3,>=3). Sum each cell with the global per-leg denominator
so cell residual contributions add exactly to the original joint error.
Also report a symmetric within-cell versus population-mass decomposition.
No per-object measured-mask substitutions. Repeat original sums as a mandatory
replay check. Whole-case paired bootstrap, 10000 resamples, seed20260925.

Maximum two simultaneous GPUs; CPU catalogue processing also uses Slurm.
Four constant cases are localization diagnostics, not independent validation.

## Component-estimator audit

Before interpreting population transport, re-reduce the saved standalone flow
sufficient statistics as R11 and R22 separately. The old standalone row was a
half-trace, whereas constant shear tests R11. This checks whether cancellation
between the two diagonal errors hid a component-specific discrepancy. No new
model evaluation is needed; use the same paired case bootstrap and nuisance
recomputation as the saved standalone report.

## Neighbour-shear step comparison

The existing training simulations also measured primaries on the h=0.05 leg.
Compare h=0.2 and h=0.05 on identical neighbour pairs and primaries usable in
all three legs, retaining bright-neighbour exclusions in all legs. The baseline
geometry, emulator prediction and selected baseline denominator are shared.
Verify the two sheared truth catalogues have identical identities and shear
directions, with magnitudes differing by exactly four. Compare both baseline
shape response and full selected response. This tests finite-step transport
without generating new images or changing populations between amplitudes.
It remains a forward random-direction experiment, not coherent +/-0.02.

## Standalone flow in matching truth cells

After the joint map showed its residual concentrated in true radius0.15–0.4
and nbr_close_100<0.1, re-evaluate the same 40 held-out flow cases with original
16 draws/seeds. Save the existing response sufficient statistics per fixed
truth cell as well as per case. Use the same global measured baseline mass,
mean and shear normal, so cell errors sum exactly to the original hard-cut
R11 error. Compare locations and cancellations, not a direct additive budget
across training and constant-shear populations. Replay the soft and hard global
responses against saved results before interpreting the map.

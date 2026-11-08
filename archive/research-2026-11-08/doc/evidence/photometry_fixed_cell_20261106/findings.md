# Why generated photometry misweights neighbour response

## Four-scene fixed-photometry result

All four workers and reducer17807065 completed exit0; every original-moment,
bin-membership and bootstrap replay check passed. The [full report](results.md)
finds a coefficient contrast of **+0.006547 +/-0.000172** with original photometry
and **+0.006547 +/-0.000162** after radius and magnitude are made exactly identical
within each bin. The within-bin term is +0.005938 +/-0.000242 originally and
+0.005929 +/-0.000244 after fixing photometry. All four scene contrasts remain
positive. These uncertainties describe paired model contrasts across four fixed
development scenes, not independent calibration accuracy or training uncertainty.

The response mismatch therefore survives removal of residual sub-bin photometry
differences. With identical emulator photometry, the generated selection weights
still pick primary/neighbour configurations with lower average response. Scalar
coefficients do not depend on generated ellipticity, so conditional shape
transport cannot explain this contrast.

## Selection depends on detailed neighbour response

The [selection audit](selection.md) evaluates each truth scene at one reference
radius3.5/magnitude24.5 and sums its modelled neighbour coefficients. This score
is a diagnostic of the scene, **not measured or calibrated R11**. Overall pass
fractions agree closely: generated0.476202 versus measured0.477002. But for
reference coefficients0.2--0.4 the fractions are0.449123 versus0.461445; for
0.4--0.8 they are0.482632 versus0.496031. Generated photometry underselects the
high-response configurations.

The measured-minus-generated selected mean reference score is
+0.003599 +/-0.000078. Its within-stratum component remains+0.003404 +/-0.000325
after coarse intrinsic size/magnitude/Sersic controls, and+0.003401 +/-0.000322
after adding close050 crowding. Those strata do not hold all eight flow inputs
exactly fixed and do not prove which missing feature is responsible.

The [context audit](context_audit.json) recomputes all eight raw context values
from source truth for256 uniformly sampled targets in each scene. All1024
targets match exactly, including circularized radius and crowding units.
Training/runtime crowding formulas also agree on the geometry fixture. This
rules out the tested context-cache/units mismatch, not every training-cache row.

The completed [held-out information diagnostic](../selection_information_20261106/findings.md#four-held-out-scenes)
compares selection predictors using existing inputs, a larger-capacity version,
and an added detailed-scene response score. Their selected-mean reference-score
errors are+0.002619,+0.002584 and-0.000010 respectively; the original flow gives
+0.003599. Added information improves held-out selection log loss in every
scene. This supports richer scene conditioning as a practical direction while
retaining the common-usability and development-scene limitations. No production
flow or calibrated correction is replaced by these diagnostic regressions.

## Pilot: identical photometry preserves the mismatch

V1001000 job17807045 and reducer17807054 completed exit0. Original per-bin mass
and coefficient numerators replay, fixed representatives stay in their original
bins, target moments reproduce histograms, and every original bootstrap
coefficient reproduces the preceding report. The [pilot](pilot_results.md)
finds an original within-bin coefficient difference0.005567 and a difference
0.005609 after fixing radius and magnitude exactly. Total contrasts are0.006056
and0.006102 respectively. These are one-scene model contrasts, not simulation
calibration errors or population significance estimates.

Residual photometry differences inside the bins therefore do not explain this
pilot result. With identical emulator photometry, the selected objects still
have different average response coefficients. That difference must enter through
the fixed truth-pair features and their weights: primary properties, neighbour
properties, geometry/counts, and their association with which photometry is
selected. It is not caused by conditional flow shapes, which do not enter the
scalar-coefficient calculation.

The three remaining original scenes subsequently passed the same strict controls;
the complete results and supplementary selection audit are reported above.

## What this can and cannot identify

The flow sees primary intrinsic properties and three scalar flux-crowding
summaries. The response emulator additionally uses each neighbour's size,
magnitude, Sérsic index and separation, together with the primary's physical
properties and output photometry. Thus the composition averages a function of
photometry and detailed scene properties, not merely a function of the marginal
photometry distribution. Accurate counts or means do not guarantee that average.

Two explanations remain: the flow has not learned the dependence on its current
inputs accurately enough, or those inputs omit scene information needed to retain
the photometry/response association. The current intervention identifies where
the mismatch enters but does not distinguish those explanations causally.
No changes to the emulator weights or fitted response offsets are justified by
this diagnostic alone.

## Complementary primary-versus-neighbour image replay

The owner's [grid replay proposal](../primary_neighbour_grid_20261106/design.md#four-shear-corners-and-exact-closure)
would isolate the primary response from all neighbours rather than random half-
groups. Its four-corner estimator passes analytical closure fixtures including
changing selected denominators. A coherent grid-to-original-image control and
explicit source sampling/optimizer-noise design are needed before a large
campaign; those are distinct from the algebraic sum identity. The current work
continues to use existing catalogues while that operational pilot is designed.

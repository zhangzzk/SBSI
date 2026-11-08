# Fixed-photometry response audit

## Question and fixed sample

All four scenes confirm a dominant within-radius/magnitude-bin response
difference. Binning alone does not exclude residual photometry differences
inside each bin. Freeze the emulator photometry to a common representative in
each fine bin while preserving the original branch's bin membership, selection
weights and all truth-pair features. This tests whether the within-bin effect
survives exactly identical emulator photometry.

Use the weak-nine flow, original emulator, scenes1000/1040/1080/1120, common
actual usability, radius>3 pixels/magnitude<25, original full-parent1024 blocks,
64 portable antithetic draws and original GPU families. No image simulation,
training, model substitution, or empirical correction. Broad17788286 stays held.

## Estimator and intervention

Retain generated/generated and measured/measured photometry/selection branches
from the preceding factorial. The measured branch averages endpoint masks and
half-contributes each endpoint, including cut-crossers. Analyze the mean scalar
opposite-group emulator coefficient, which accounted for0.006547 of the0.006556
finite shape-response shift. No conditional-shape transport is needed here.

Fine bin edges are unchanged from photometry_response_bins_20261106. For finite
positive radius bins use geometric centres; the0–1 bin uses0.5 and the20–infinity
bin uses25 pixels. Magnitude bins use arithmetic centres; open tails use19 and28.
These fixed representatives are diagnostic interventions, not corrected model
outputs. Selection and bins are always calculated from the original outputs.

Save per-bin mass, original coefficient numerator and fixed-photometry
coefficient numerator. Original mass and coefficient must replay the previous
histogram in every bin with rtol2e-10/atol2e-7. Every representative must map
back into its original bin. Decompose original and fixed coefficient contrasts
into distribution, within-bin and exclusive-support terms using the original
whole-case bootstrap and its coarse-resolution controls. Report their paired
difference and casewise values; add delete-one-scene SEs.

Also preserve per-target moments and raw flow context for later association
tests. A single reference-photometry coefficient at radius3.5/magnitude24.5
records response sensitivity to each object's truth scene, without fitting it.
These auxiliary data do not enter the primary estimator or select a model.

## Reference-response selection stratification

Use the recorded coefficient at radius3.5/magnitude24.5, summed over both
neighbour groups, to stratify selection before inspecting the stored target
moments. Fixed coefficient edges are-infinity,0,.025,.05,.1,.2,.4,.8,infinity.
For each primary, average generated pass probabilities over the two own-shear
signs and measured pass indicators over the four image states. Report measured
minus generated pass rates and selected-population fractions by reference-
coefficient interval. The coefficient is a scene sensitivity score at common
photometry, not a new calibrated R11 measurement.

Also decompose its selected-mean difference within fixed intrinsic cells:
circularized Re edges.15/.25/.4/.7, intrinsic magnitude23/25/27, Sérsic n1/2.5.
A second partition adds the existing nbr_close_050 context at.05/.2/.7/1.5.
These are coarse controls, not exact conditioning on all eight flow inputs.
Preserve exclusive support and case uncertainty; do not claim omitted-feature
causality from this descriptive association alone. No learned correction or
new model is fitted.

## Interpretation and resources

If a within-bin difference survives, the emulator is averaging different truth
configurations at identical photometry. Target intrinsic properties and detailed
neighbour properties remain possible sources. This alone cannot distinguish
incorrect learned conditional distributions from missing conditioning features.
Coarse-bin controls aggregate the original fixed fine-bin representatives; only
the fine-bin result has exactly identical photometry within every reported bin.

A V100 case1000 pilot and CPU replay gate the other three cases. Maximum three
workers after the pilot, below the six-GPU ceiling. Scheduler audit finds no
other running study GPUs. Login-node resource detection is informational only;
all catalogue-scale computation runs in Slurm. No new image campaign is started
by this protocol; the owner's proposed scene replay is designed separately.

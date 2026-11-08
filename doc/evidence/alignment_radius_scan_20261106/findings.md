# Cartesian radius scan

## Scope and status

[Protocol](rule.md#question-and-fixed-settings) follows the completed
[axis comparison and exact cancellation decomposition](../alignment_axis2_constant_20261105/findings.md#exact-radius-partition-exposes-cancellation).
Frozen seed611, original C8/emulator, both axes,40 existing scenes. No retraining,
image changes, or selection of a new threshold by measured bias.

Pilot17785984 tests case1000 on both axes and requires all six existing cuts to
replay after regrouping into intrinsic-size bins. Remaining78 tasks17786000
wait for both pilots; maximum six GPUs. Pilot report17786001; full17786002.
Two-hour bounded check17786134 records status without waking the chat.
Both pilots reached finite-sample accumulation through block300 at startup.
Full result pending; no inference from single-case bias. Pilot case uncertainty
is explicitly null because there is only one independent scene.


## Pilot replay and full-stage startup

Both pilot tasks17785984 completed0 in5m12s. All six overlapping selections
match prior model/simulation sums under the declared roundoff tolerances;
reducer17786001 completed0 in4s and reports null case SE for the single scene.
Array17786000 began six concurrent workers across cip/inter; all six reached
finite model accumulation (PROGRESS block0 or later), verified before yielding. Final scientific
conclusions await the all40-scene reducer; pilot values are not calibration
claims. [startup_receipt.json](startup_receipt.json) preserves per-axis checks.


## Stopped after cross-device replay failures

Later tasks failed the model-sum replay when scheduled on a different GPU
family from their old cached evaluation (e.g.case1083,g2: A40-8Q versus RTX2080Ti).
Twelve failures were recorded before the stop; no complete40-case scan result.
Remaining/running array17786000 and full reducer17786002 were cancelled;
all successful and failed receipts are preserved. Subsequent explicit device
test confirmed CUDA RNG produces different1024-row latent banks at the same
seed. Portable CPU banks make test draws bitwise equal across both GPU families.
Continuation prioritizes the owner's [nine target cuts](../alignment_target9_20261106/rule.md#device-independent-sampling-amendment-before-baseline-execution),
using a fresh shared realization. This is a sampling reproducibility issue,
not evidence of a new physical response bias.

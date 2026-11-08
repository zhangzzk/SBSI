# Response weighting within photometry bins

## Question and fixed comparison

Continue the authorized neighbour-response localization, without resuming the
broad regularization evaluation. Use the same weak-nine checkpoint, emulator,
four scenes, common actual usability, target radius>3pixels/magnitude<25,
original64 portable draws and hardware families as the preceding factorial.

The next diagnostic distinguishes a different distribution of radius/flux from
a different mean response within the same radius/flux bins. This is a descriptive
decomposition, not proof that the flow lacks particular conditioning features.
Intrinsic properties and detailed neighbour configurations can differ within a
bin. Inspect multiple resolutions, with and without intrinsic-size strata.

## Exact identities and controls

Retain generated/generated and measured/measured diagonal branches only. For
the measured branch, assign each endpoint's half contribution to its own
photometry bin while retaining the preceding endpoint-average selection weight.
Thus rare objects straddling a cut can contribute outside that cut; keep those
bins and do not silently drop them. Shapes remain conditional flow draws.

Store selected mass, finite shape-change numerator and scalar neighbour-response
coefficient numerator. Summing bins must replay all corresponding factorial
per-edge, per-size raw moments with rtol2e-10/atol2e-7. Same original draws and
model identities; no response offsets. A pilot and reducer gate remaining cases.

For each edge and common-support bin, decompose the model mean change into
delta(p)*average(conditional mean) and average(p)*delta(conditional mean).
Keep exclusive-support contributions separate. These terms must sum exactly
to the diagonal response change for every whole-case bootstrap resample.
The four opposite edges retain their separate normalization and equal weights.
No doubling into the global coherent response budget. Compare fine, medium and
coarse bins; report case uncertainty and Monte Carlo limitations from the parent
factorial. This experiment cannot establish conditional density calibration.

## Execution

Pilot case1000 on V100; remaining1040/1080/1120 on A40/V100/RTX2080Ti only after
strict replay and reducer gates. At most three remaining workers, with combined
study use at most six GPUs. Held array17788286 remains held. No image generation,
training, new classifier or emulator, or empirical correction.

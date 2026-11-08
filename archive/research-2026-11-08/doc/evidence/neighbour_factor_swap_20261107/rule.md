# Fixed photometry/conditional-shape factor swaps

## Question and endpoints

The selected-neighbour constraint improves its target on own data and on
constant-shear scenes, but an opposing change in the flow contribution
outweighs the emulator-contribution increase. Separate two mechanisms:
changes to the generated photometry/selection distribution, and changes
to the conditional shape density at fixed photometry. Apply the same
diagnostic to original NLL-to-nine-cut regularization and to the added
neighbour moment. No training, loss tuning or endpoint selection is done.

The density factorizes as p(photo|x) p(shape|photo,x). All three frozen
sources have identical context/target standardizers and outer transforms.
Retain their three full products and assemble the four cross-products in
manifest.json. Source fingerprints and every factor's provenance are fixed
before any evaluation. Hybrids are interventions for localization, not
promoted models. A full model can be wrong even if one diagnostic improves.

## Correctness gates

Compare all source parameters/buffers outside the two factors exactly.
Require matching condition/target/config states. Verify hybrid photometry
equals its photometry source exactly, conditional-shape log density equals
its shape source at fixed measured output, and saved/reloaded outputs and
densities agree exactly. Reassemble each unchanged source as an identity
control. Record fixed own-data NLL and original full-matrix response probes
for every product using the previous rows, streams and cuts.

The initial verification attempt exposed float32 addition of the photometry
and shape log determinants in the existing joint implementation. Direct
subtraction of joint minus marginal density differs by up to9.54e-7 across
products even when the conditional factor is identical. Evaluate the
conditional factor directly and require exact equality; reconstruct the
original joint density including the explicitly computed float32 addition
roundoff and require agreement within1e-10. This fixes the verification
estimator, without changing model arithmetic, predictions or density code.

For constant-shear evaluation, evaluate all seven products together on each
worker with identical portable draws and original1024-parent blocks. This
gives a paired decomposition on the same GPU. Reuse the validated pooled
emulator tree lookup; require first-block agreement to separate calls at
2e-14 absolute/relative tolerance. Models sharing a photometry factor must
have exactly equal selections and selected-reference-score sums. Compare
the three unchanged source endpoints to prior case outputs, requiring all
response components to agree within1e-6 absolute (float32 cross-device audit,
declared before evaluation; not a correction or error-bar inflation).
Keep exact measured-selection masses, reference-score and simulation replay
checks. Preserve all original per-leg populations and catalogue matching.

## Decomposition and execution

Let A be the old endpoint and B the new endpoint; BA uses B photometry and
A conditional shapes, and AB uses A photometry and B conditional shapes.
For every cut, axis, flag mode and response contribution, report:

- photometry effect at old shapes: BA-AA;
- shape effect at old photometry: AB-AA;
- interaction: BB-BA-AB+AA;
- total change: BB-AA.

Verify that these terms sum exactly within roundoff. Also retain shape
effects at new photometry and photometry effects at new shapes. These are
model intervention effects, not standalone component errors. Use paired
whole-case bootstrap/jackknife errors, four reused development scenes, both
axes, all nine target cuts and the original auxiliary cuts. Definitions are
[CONVENTIONS response estimators](../../CONVENTIONS.md#6-response-estimators)
and [multiplicative bias](../../CONVENTIONS.md#7-multiplicative-bias).

One GPU allocation assembles/checks products, evaluates own-data probes and
runs the first case/axis. Correctness gates release three persistent GPU
workers for the remaining seven case/axis tasks; each keeps all products
and the emulator loaded. A dependent CPU reducer makes the factorial report.
Do not restart the held broad evaluation. Keep the trial within three GPUs
and the study within six; use the existing two-hour same-thread loop for
long-run follow-up. No new images, empirical offsets or production changes.

The launcher verifies each stage against `source_pins.json` and compiles the
verified source bytes directly. This avoids an observed stale timestamp
bytecode reuse on the compute node. Each executed stage prints its digest;
the failed preparation attempts and their source/output records are retained.

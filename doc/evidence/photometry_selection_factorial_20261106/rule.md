# Matched photometry/selection factorial

## Fixed question and sample

The owner requests continued localization after measured photometry/selection
restores the neighbour response. Freeze the weak nine-cut checkpoint and all
four scenes1000/1040/1080/1120. No training, new images or model replacement.
Use original classifier/emulator identities and the common-all-four actual
usability cohort. Target radius>3pixels and magnitude<25; also retain the
original uncut/magnitude-only/radius-only diagnostics. All errors are simulation
minus model and use within-group opposite-shear responses, equally averaged
over the A/B groups. Do not double these into a coherent full-population error.

## Four crossed input choices

For each opposite-group edge, predict its finite disk-transport shape change
using (G) generated size/flux with its own conditional flow shapes or (M)
measured endpoint size/flux with conditional flow shapes from the same model.
Average the two measured-photometry endpoints as in the preceding conditional
test. Cross each choice with generated selection weights or the arithmetic
average of the actual endpoint selection masks. Each branch uses its own
declared average selected denominator. The generated weights remain attached
to the same latent draw, preserving their actual shape/photometry association.
Only the shape-response component enters this factorial; measured selection
response is retained separately and is not implicitly fitted or corrected.

Report all four means and both intervention orders. The symmetric input-value
and membership effects are one-half of the sum of their two corresponding
edge differences; they must add exactly to the diagonal G/G-to-M/M difference.
Also report the interaction. These are controlled model substitutions, not
unique physical error components. A measured selection mask on an independent
flow draw deliberately removes a real measurement association; do not call it
a corrected generative prediction.

## Controls, uncertainties and execution

Use original full-parent blocks of1024 and64 portable antithetic latent draws,
seed7301+10000019*case+1000003*block. Draw the generated branch on the full block
before restricting to common usability, so it replays the original stream.
Conditional branches use the identical shape latents and fixed target-own-shear
context. Verify conditional reconstruction at generated photometry and preserve
measured photometry exactly. Generated/G weights and numerator must replay the
existing weak-nine four-state common-U per-group moments with rtol2e-10 and
atol2e-7. Use each case's successful GPU family: V100,A40,V100,RTX2080Ti.
No numerical tolerance adjustment or empirical correction is authorized here.

Compare the measured/ M endpoint with the earlier conditional estimate and
report its change from the different draw bank; retain draw-half estimates
to expose Monte Carlo sensitivity. Resample whole cases with10000 shared
bootstrap replicates, seed202611069301. Fixed five intrinsic-size bins remain
diagnostic only. Every factorial identity is checked in every resample.

One V100 case1000 pilot and CPU replay/reducer gate the remaining three GPU
workers. At most three factorial workers after the pilot, alongside the one
older baseline worker if still active; combined study use stays below six.
Broad array17788286 remains held. Save source/checkpoint hashes and unmatched
catalogue receipts. Follow startup and then use a bounded scheduled check.

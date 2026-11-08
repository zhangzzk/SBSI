# Freeze the nine-cut conditional shape model during photometry training

## Paired optimizer intervention

Both runs start from the same fitted neighbour-conditioned nine-cut control
(checkpoint 6fc0c373… in manifest.json), before the added neighbour-moment loss.
Keep the entire conditional shape factor, outer transforms, standardizers,
classifier and emulator fixed. Only parameters in base.base.photometry enter
AdamW. Keep autograd through the frozen shape function with respect to its
photometry inputs for the original nine-cut response loss. The two arms differ
only in the weight (0 / 0.001) of the original selection-mass and selected-
neighbour moment; likelihood, full-matrix response weight, training rows, draws,
seeds, clipping, rates and 64 updates are as in the prior experiment.
Targets are the verified original training measurements. The zero-weight arm
controls for extra training. The untouched nine-cut start is provided separately.
This is a new 64-step continuation, not an exact historical free-vs-frozen
training ablation from the NLL initializer.

Verify that every state tensor outside photometry is identical to the start on
completion and reload, gradients are absent there on every update, some
photometry weights change, and direct conditional density at fixed observed
photometry and contexts is identical before and after. Preserve per-eight-step
optimizer checkpoints and require the prior experiment's source and target hashes.
Compare the held-out 500k-row hard-cut score error and pass rates, standalone
response probes, and four fixed development scenes on both axes and all nine
cuts. Case-paired differences use common draws. Separate flow and emulator
contributions, predicted versus actual usability and measured-cut intervention.
The measured-cut arm still uses generated photometry for the two predictions.
No outcome-based choice of endpoint, loss weight, or change in public models.
Prior work: [frozen swaps](../neighbour_factor_swap_20261107/scene_size_breakdown.md#central-residual-and-component-changes).

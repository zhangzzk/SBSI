# Cartesian factorized flow: predeclared seed replication

## Scientific choices

Add seeds612 and613 to the existing seed611 Cartesian model. Change model
initialization and training sampling/order seeds only. Freeze original160/40
cases, preprocessing,140epochs x4million rows,8192batch, AdamW, FS2 realized
learning rates, and500k validation rows. Select minimum validation NLL only.
No response penalty or checkpoint/model selection using constant-shear results.
Keep physical maps, architecture, classifier and emulator unchanged.

Run original standalone validation on both additional selected checkpoints,
then all40 existing constant-shear scenes, same64-draw CRNs and per-leg cuts.
Report each seed separately and paired differences. Three training seeds give
only a rough estimate of training variability; never divide model spread by
sqrt(3) and call that a calibrated single-model uncertainty. Do not select the
best seed by development m. No new acceptance claim from reused scenes.

## Resources and validation

Reduce the existing matched-control array to four workers; two seed-training
workers start only after all recorded presently running control allocations finish.
Then two standalone workers depend on both training tasks. Six GPUs maximum
across this study. Constant-shear replication waits for the matched controls
and both standalone jobs, so its own six-worker cap cannot overlap them.
Use the installed frozen F8 runtime and explicit helper imports. Validate finite
losses, exact checkpoint reload, checkpoint SHA from completion, measured replay,
and all40-case paired reports. Keep all outputs separate from retained models.

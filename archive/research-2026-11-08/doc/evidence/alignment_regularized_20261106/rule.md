# Does one-cut or nine-cut regularization preserve Cartesian performance?

## Fixed comparison

Owner asks whether the likelihood-only improvement survives response penalties.
Start every arm from the identical factorized Cartesian seed611 checkpoint.
Five fixed continuations: NLL-only control; one-cut weights0.001 and0.01;
nine-cut weights0.001 and0.01. The one cut is radius>3pixels,mag<25. The nine
cuts are radius>2.8/3.0/3.2 AND mag<24.9/25/25.1. All four response-matrix
components enter individually, equal mean over cuts/components. No half-trace
loss and no global or single-axis cut additions.

Each arm has64 updates, AdamW lr1e-5, weight_decay0, NLL batch8192. Same NLL
row stream and same guard row/latent streams across arms. Per step the guard
uses two independent32768-pair groups,4 antithetic draws,1024-pair replay chunks.
Combine NLL and weighted guard gradients before one norm5 clip and optimizer
update. Guard scale per cut is0.003 times the measured training half-trace;
this normalizes all four components without averaging their residuals. Smooth
selection widths0.05pixels/0.05mag. This is a bounded continuation experiment,
not a claim that every strength or converged regularized solution is tested.

## Targets, checkpoints and diagnostics

Compute exact measured guard targets from all pairs in the160 flow-training
cases. Use the existing matched-usable forward-shear estimator on both sides;
this is a flow-alone proxy, not the coherent end-to-end constant-shear target.
Validation cases, coherent scenes and joint biases never provide training
targets. Preserve original C8/emulator and measured catalogues.

Use fixed step64 endpoints for every comparison, regardless of measured bias
or validation score. No winner selection among checkpoints/arms. Measure common
500k-row validation NLL and fixed262144-pair train/validation response probes
before/after (two16-draw latent groups, paired across models). Their finite
pair/latent integration uncertainty is not a case-bootstrap uncertainty; do
not interpret tiny probe changes alone as physical improvement.

Verify independent-bank replay backward against direct autograd and exact
checkpoint reload. Preserve clipping/gradient histories and matching receipts.
Evaluate every endpoint on both constant-shear directions, all40 existing
scenes, all nine target cuts, same64-draw CRNs and per-leg normalization.
Compare with original NLL-only checkpoint and matched NLL continuation using
paired whole-case bootstrap; report each component/cut, RMS and max bias.
The central question is preservation and local robustness. Internal cancellations
and distant-cut errors remain diagnostics, not automatic failure criteria.

## Staging and resources

Prepare arrays/targets on scheduled CPU nodes. Restrict the existing broad
scan to three GPUs and run at most three continuation workers after all
currently active scan tasks drain. The queued nine-cut baseline additionally
waits for training completion; its six-GPU array cannot overlap training.
New constant evaluations follow the baseline and training, at most six GPUs.
First case/axis evaluation gates the full evaluation array. Keep output roots
separate and preserve model preset identities. Follow startup and use bounded
scheduled checks for long stages.


## Startup corrections and portable evaluation

Preparation infers16 transformed features from the preprocessor rather than the
8 raw columns. Final train allocations use A40-16GB,24GB host RAM; the prepared
arrays and all gradient checks fit. Guard-training and probe banks use the same
CUDA device family in every arm. Direct versus replay gradient errors are~1e-7.

The older broad scan failed strict cross-device random-draw replay and was
stopped. The independent resource-drain prerequisites are satisfied; training
now waits only on CPU preparation. No ongoing broad scan shares GPUs, so the
obsolete release job17788490 was cancelled. Exact-grid baseline waits on all
training AND portable-sampling verification17789022.

Use the [verified PCG64 banks](../alignment_target9_20261106/rule.md#device-independent-sampling-amendment-before-baseline-execution)
for all constant-shear baseline/endpoint comparisons. These are a new Monte
Carlo realization, explicitly compared with the old baseline in the reducer.
No model offsets or scaling changes; the earlier CUDA realization is preserved.

# Target-cut robustness

## Owner correction and current objective

Internal cancellations are acceptable for the intended sample. The immediate
objective is stable response around radius3pixels and magnitude25, using the
nine joint cuts in [the protocol](rule.md#owner-objective). The broader0.75-arcsec
failure and size-bin cancellation are useful diagnostics, not grounds alone to
reject this intended selection. Preserve those measurements without treating
uniform subpopulation accuracy as a prerequisite.

The improved Cartesian checkpoint is NLL-only. Earlier nine-cut regularization
used the older full-dimensional flow and different magnitude spacing; its
outcomes do not establish whether local regularization helps this architecture.

## Exact-grid baseline and planned penalty

Evaluate unchanged Cartesian seed611 on both constant-shear axes and all40
existing scenes, predicted/true usability and generated selection. Nine target
cuts plus five replay-only selections; all six overlapping legacy selections
must reproduce measured simulation sums. Added paired RMS, maximum absolute
bias and range over the nine cuts. Correlated cuts use the same case-bootstrap
weights. Model-sampling changes follow the amendment below.

Pilot17787232 waited on completed regularization training17788119 and portable
sampler verification17789022. The broad scan was cancelled after cross-device
CUDA RNG replay failures, and its old dependency removed. Remaining17787261
depends on both pilots; reports17787262/17787293 and a bounded2h status check
17787294 are queued. Syntax and exact9-cut product checks pass. Axis2 pilot
completed exit0 in5m13s; axis1 is progressing normally. Full40-case scientific
results are pending. All images/models already exist.

The next regularized training recipe should use the160 flow-training cases,
NLL plus the nine separate full-matrix response errors, independent replica
pairs for gradient estimation, and validation around the intended cuts. No
coherent-test target fitting, half-trace-only objective or empirical offset.
The follow-up [matched continuation study](../alignment_regularized_20261106/rule.md#fixed-comparison)
now fixes two strengths and64-step endpoints and queues training. This baseline
evaluation still uses the original checkpoint.

## Verified portable sampling

[Device verification](device_verification.json) confirms that CUDA normal banks
with identical seeds can differ between A40 and RTX at1024 rows. NumPy PCG64
CPU banks, cast tofloat32 and paired antithetically, produce identical latent
hashes and bitwise-identical physical draws on both GPUs at1024 and528 rows.
All new baseline and regularized comparisons use these shared portable banks.
This is a new random realization, explicitly distinguished from old CUDA draws;
the full reducer reports paired differences from the old realization on six
overlapping cuts. Measured-simulation replay remains required. No numerical
tolerance was loosened to hide the earlier stochastic mismatch.

## Completed 40-case baseline

Both pilots, all78 remaining tasks and full reducer17787293 completed exit0.
The [full report](full_report.json) uses all40 cases and both shear axes. With
predicted usability and generated selection, radius>3pixels and magnitude<25:
m1=-0.25517% +/-0.30185 percentage points; m2=-0.51535% +/-0.24713 percentage
points. These are paired whole-case bootstrap standard errors conditional on
fixed models. Definitions follow [Conventions](../../CONVENTIONS.md#7-multiplicative-bias).

Across the nine target cuts, central m1 values range from-0.43782% to-0.02083%;
m2 ranges from-0.51977% to-0.02515%. Nine-cut RMS is0.28924% and0.37336%,
respectively; overlapping cuts are correlated. True usability with generated
selection gives central-cut m1=-0.05021% +/-0.30311pp and
m2=-0.32750% +/-0.24665pp. These masks still do not impose measured selection
on independent model draws.

The portable latent realization changes central predicted-usability bias by
+0.01876 +/-0.00800pp on axis1 and+0.02003 +/-0.00935pp on axis2 relative to
the older CUDA realization. This measured change is small compared with the
absolute case uncertainty and is preserved explicitly. Every regularization
comparison uses the new shared realization. No independent acceptance claim.

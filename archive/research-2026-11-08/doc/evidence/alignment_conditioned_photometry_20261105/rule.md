# Conditional shape at measured photometry

## Predeclared scope

Freeze the two factorized Cartesian/frame checkpoints from the preceding study.
Use all40 existing GalSBI standalone validation cases, the original seven cuts,
forward two-component response estimator and original two16-draw antithetic
banks with identical four-dimensional latent streams and case/block seeds.
Keep each leg's exact measured radius and flux and draw from the trained
conditional shape distribution at those values. No retraining or new images.

Compare to the saved ordinary joint draws on the same cases. Absolute response
error is simulation minus model; alignment error is model minus simulation.
Report full2x2 response and case-paired10000-bootstrap uncertainties, intrinsic
size contributions and alignment profiles. Fixed-model development diagnostic,
not independent constant-shear acceptance. Changing the conditioning also
changes the estimand; a difference is not an exact additive physical bias budget.

## Scientific distinction

Actual membership times an independent joint draw estimates S(y_obs) E[e|x].
It removes measured shape-photometry dependence. This test instead estimates
S(y_obs) E[e|radius_obs,flux_obs,x]. It tests the conditional shape factor under
the observed photometry distribution, while exactly preserving measured cuts.
The standalone target-shear experiment avoids interpreting measured constant-
shear photometry plus a neighbor emulator as an unambiguous causal decomposition.

## Checks and resources

Reconstruct joint generated shapes from their generated photometry and the same
shape latent; verify unused photometry latents have no effect, exact CRN across
identical legs, exact observed photometry, physical support, measured-case replay,
exact per-draw membership and per-cell selection masses; cross-term sums
allow1e-12 floating accumulation tolerance. Measured replay uses the existing
1e-12 relative/1e-8 absolute tolerance for GPU sums. Reject nonpositive photometry
without clipping or silently dropping cases. One GPU per arm, two concurrent
workers on cip/inter within the owner-authorized six-GPU study ceiling.

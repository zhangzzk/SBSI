# True usability and final measured selection, 2026-11-05

Owner requested the absolute-error table using true selection and detection flags.
Interpretation: per-leg true usable flag U times that leg's measured radius>3px
and MAG_AUTO<25 mask S, applied identically to simulated and model contributions.
Replace generated selection; do not apply both masks. Keep original generated
shape, radius and flux for the model and its emulator evaluation. No classifier
probabilities enter the oracle U*S row.

Use the four previous matched cases1000,1040,1080,1120, original64 draws and seeds,
original1024-row blocks, older16-cut GalSBI-dev model. Replay all four previous
substitution branches and validate against their saved sums before reporting the
new flags-only branch. Maximum two concurrent GPUs. Also reduce original predicted
U/generated S and true U/generated S on exactly these cases for comparison;
retain the earlier140-case numbers only as explicitly labelled historical rows.
10000 paired whole-case bootstrap resamples, seed20260925, fixed model/draws.

The flow and emulator contributions on constant-shear data are model contributions,
not component errors: the simulations provide only their combined true response.
Standalone component checks use their own training-family populations. The emulator
check already uses actual usable labels and measured-g0 cuts. An oracle object mask
does not make untruncated generated outputs samples from the measurement distribution
conditional on passing the cut; the result is a diagnostic, not pure classifier or
selection bias, and not a deployable correction. No training or model changes.

## Standalone flow (owner clarification)

Owner explicitly requested recomputation on its own data as well. Use all40
held-out flow-training-family cases and original16 antithetic draws/block seeds.
Compute both the original soft-cut ruler and hard radius>3px/MAG_AUTO<25. For each,
compare generated per-leg selection to measured per-leg selection multiplying the
untruncated model shape mean. All pairs are actually usable in both legs, as in
the existing standalone check. The response is the half-trace of the original
forward random-direction influence regression. Simulation and model share the
measured baseline selected mass and mean. Save case sufficient statistics and
recompute mass, mean and shear normal matrix in each case bootstrap. Replay the
original soft-cut simulation/model point responses before interpretation. Schedule
after the joint array to keep the combined maximum at two GPUs.

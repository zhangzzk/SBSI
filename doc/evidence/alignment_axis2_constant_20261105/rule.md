# Constant-shear second-component test

## Predeclared experiment

Use the same40 scene IDs and galaxies as the completed g1 constant-shear study.
Replace (g1,g2)=(+/-0.02,0) with (0,+/-0.02). Keep positions, morphology, flux,
PSF, pixel/noise conditions, renderer seed1000*case, detection and centered65px
shape measurement unchanged. Original files are read-only; new products belong
to BlendEMU under constant_axis2_20261105. Same scene/noise seeds across axes
permit paired comparison but do not make the new measurements identical.
Original ngmix initialization is unseeded; preserve the measurement algorithm
and disclose fitting randomness. Do not change PSF fit, circularized size or band.

Freeze seed611 factorized Cartesian and retained GalSBI flow, original C8 and
physical-moment emulator. Generate C8 probabilities at g2+/-0.02 from the original
parent contexts; old g1 probabilities cannot be reused. Use same64-draw streams,
per-leg measured/model cuts and Mobius composition with a g2 shear vector.
Report R12/R22, m2=100(Rsim22/Rmodel22-1), absolute error, each flag mode, paired
axis/arm comparisons. No checkpoint fitting or selection with these outcomes.
This tests another shear direction on existing scenes, not independent-population
acceptance. Seed612/613 replication is a separate frozen g1 test.

## Pipeline gates and resources

Prepare all40 input pairs with exact non-shear-column equality and config-only
path substitutions. First four scenes (eight legs) form an operational pilot;
remaining36 scenes depend on successful pilot measurements and direction checks,
not on the measured bias. Reuse the completed unchanged-input case1000 render
control only after confirming renderer/measurement source hashes match its study.
Verify output truth g1=0,g2=+/-0.02, all non-shear truth columns vs original
(tolerance5e-16 only for derived e1/e2), catalogue existence and finite usable
measurements. Retain unmatched-row diagnostics in the final evaluator.
Eight single-worker pilot jobs. On2026-11-06 the owner authorized up to500
CPUs. Pack the72 remaining legs into nine allocations of eight CPUs each,
with eight independent singleton workers per allocation. All72 can therefore
run concurrently within the50-job/user limit. Preserve the pilot dependency,
original per-leg launcher, seeds and science settings. Extra CPUs beyond72
cannot accelerate the unchanged case-parallel pipeline. No GPU for simulation.
GPU evaluation waits for seed-replication GPU stages to finish; six-worker cap.
Check startup and use bounded scheduled status checks for these long jobs.

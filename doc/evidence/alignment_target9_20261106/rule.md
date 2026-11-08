# Target-cut robustness for the improved Cartesian flow

## Owner objective

On2026-11-06 the owner clarified that internal cancellations are acceptable;
the objective is stable calibration around the intended selection, not uniform
accuracy in every galaxy subpopulation or at distant cuts. The relevant grid
is the Cartesian product of measured radius>2.8/3.0/3.2 pixels (0.56/0.60/0.64
arcsec) and MAG_AUTO<24.9/25.0/25.1. Exactly nine joint cuts, with equal weight.
The0.75-arcsec diagnostic is not an acceptance veto for this intended selection.

## Baseline before regularization

The improved factorized Cartesian seed611 was trained on NLL only. First
evaluate its local nine-cut robustness on both completed constant-shear axes,
all40 existing scenes, unchanged C8/emulator, generated selection and either
predicted or actual usability. These are validation diagnostics, not training
targets. Keep original64 antithetic draws, common random numbers, own per-leg
normalization and existing checkpoint. Five extra evaluation cuts replay old
sums; they are not extra regularization terms. Six-cut replay checks apply to
each case/axis. Report each response component and cut, plus worst absolute
bias and RMS over the nine cuts. Bootstrap cases jointly across all cuts;
overlapping cuts are not independent observations.

## Regularization direction

Next training work should continue this improved flow with NLL plus a smooth
selected-response penalty on the nine cuts above. Each cut has its own residual;
penalize full2x2 response components separately, rather than first averaging
signed residuals or taking the half-trace. Use response targets from the160
flow-training cases only, with their original forward-shear extraction on both
simulation/model sides. Retain the40 flow-validation cases for validation and
the coherent constant-shear tests for transfer diagnostics. No empirical offsets
or fitting to coherent-test outcomes. The old single-cut scalar recipe is not
silently reused: its half-trace constraint can hide directional errors.

The regularized trainer is not launched by this baseline protocol. Its strength,
finite-sample gradient checks, smooth/hard selection comparison and checkpoint
rule must be specified before comparing candidate validation outcomes. Model
selection should target local robustness while retaining the density fit;
response robustness away from the intended cuts remains descriptive evidence.

## Scheduling and validation

The broad radius scan already in progress remains a diagnostic. This exact-grid
baseline waits for its GPU array to finish, preserving the six-GPU ceiling.
One case on both axes is the replay pilot; remaining78 tasks depend on pilot
success. All image measurements already exist; no new image simulation.
Report fixed-model case uncertainty separately from training-seed variation.
No claim of independent acceptance from these reused development scenes.


## Device-independent sampling amendment before baseline execution

The broad scan's strict replay caught CUDA RNG differences across GPU families.
In verified case1083-sized1024-row banks, identical seed/shape gives different
CUDA latent hashes on A40-16Q versus RTX2080Ti;528-row tail banks match. TF32 is
off on both. Do not relax the failed model-sum tolerance or interpret this as a
model change. Stop the incomplete broad scan and preserve all partial outputs.

All exact-grid baseline and regularized endpoint evaluations now use NumPy
PCG64 standard-normal CPU banks cast tofloat32, then transferred to GPU. Same
64 antithetic draws, cases, block boundaries and seed formula, but explicitly a
new realization relative to legacy CUDA sampling. Reuse identical banks across
shear legs and every model arm. Verification17788989/17788990/17789022 finds
bitwise-identical new latent banks AND physical draws on A40 and RTX for both
block sizes. Evaluators require the verified sampler source hash. Original
model/checkpoint/density/scientific response definition is unchanged.

Measured six-cut sums must still reproduce exactly within summation tolerance.
Legacy model-sum differences are recorded as Monte Carlo realization changes,
not numerical replay errors. The full baseline reducer computes paired changes
from the old realization on all six overlapping cuts. All model comparisons in
this experiment use the new common realization. No empirical calibration offset.

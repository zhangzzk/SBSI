# Constant-shear test of factorized GalSBI flows

## Purpose and fixed choices

Both new flows remove the large radius2.7–2.8 alignment discrepancy; the
factorized intrinsic-frame model also improves the3.0–3.1 bin. Likelihood is
better, but standalone full-matrix response errors remain. Test whether this
transfers to the joint constant-shear bias. This is diagnostic validation,
not a declaration of production acceptance or complete causal attribution.

Freeze both NLL-selected candidate checkpoints before any constant-shear
results: Cartesian epoch139 and intrinsic-frame epoch138. Evaluate the same40
existing development cases1000–1009,1040–1049,1080–1089,1120–1129 as the earlier
grid comparison. First four cases1000,1040,1080,1120 are an interim report;
all40 are predeclared, with no stopping/model selection based on interim m.
Keep GalSBI-dev C8, physical-moment emulator, original catalogues and conditions
fixed. No response regularization, fine-tuning, offsets or new simulations.

## Prediction and flags

Use original64 antithetic draws,1024-parent blocks and latent seed
7301+10000019*case+1000003*block. Same streams for both models and both shear
legs. Constant shear g1=±0.02,g2=0; Mobius transport from original summed
neighbour emulator coefficients; K20,Rmax10arcsec over all rendered objects.
Emulator output conditioning uses each generated draw's radius/flux.
Eight original cuts: uncut,mag25.8 withr>3 andr>=3.75,mag25 withr>3,
r>3 alone,r>=3.75 alone,mag25.8 alone,mag25 alone. No truth cuts.

Report three distinct flag modes:
1. Predicted usability times each generated draw's own output selection.
2. Actual per-leg usability times each generated draw's own output selection.
3. Actual per-leg usability AND measured final selection, applied to generated
   shapes and their generated-output emulator response. This is a diagnostic
   intervention; it does not additionally substitute measured radius/flux in
   emulator inputs. Do not call mode2 true final selection.

Simulation always uses actual usability and its own per-leg measured cuts.
Drop/report unmatched catalogue entries according to the frozen measured-leg
routine. Preserve per-case sums by original intrinsic-magnitude bins. Save
flow and emulator terms with common denominators, so they add exactly to
model response; do not call the difference between simulation total and flow
alone a standalone flow error.

## Verification and reporting

One retained-flow replay case1000 must reproduce stored model/matched simulation
sums for modes1/2 and all eight cuts before launching candidates. Each candidate
case must reproduce stored measured counts/shapes. Mode3 masses must equal
measured selected counts. No Python path injection; use installed frozen F8
runtime and explicit helper-module imports. Custom factorized loader preserves
density/world-frame conventions and original checkpoint hashes.

Report absolute DeltaR=R_sim−R_model and m=100(R_sim/R_model−1), R21 residual,
flow/emulator response terms and selected-count ratios. Same10000 whole-case
bootstrap weights for all arms/modes and paired changes; preserve covariance.
Report per-case m and leave-one-case-out sensitivity, not just a small bootstrap
SE. Four-case interim uncertainty is weak; final40-case results still condition
on frozen training and measurement realizations, with no independent acceptance.
Baseline modes1/2 use exact saved sums on the same40 cases; baseline mode3 is
available only for replay case1000 and is not used as a40-case comparison.

## Resources and stages

One-GPU retained replay gates an80-task candidate array, throttled to2 GPUs.
Four-case pilot CPU report depends on its first8 tasks; full report follows all.
Bounded status audit about2h after launch. Watch startup and first meaningful
progress; use scheduled checks for the long queue. No retained identity changes.

## Resource amendment after owner request

On2026-11-05 the owner asked to use more GPUs. The existing array17768588 was
raised from2 to6 concurrent one-GPU tasks; pending tasks can use GPUs on cip or inter (excluding the old P5000 node). Existing allocations continue. Cluster partition limits are
cip3 GPUs/user and inter8 GPUs/user; the single array throttle enforces6 total.
Only scheduling/resources changed: checkpoints, cases, draws, seeds, report
dependencies and predeclared40-case scope remain fixed. See gpu_expansion.json.

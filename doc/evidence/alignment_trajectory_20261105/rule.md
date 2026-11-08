# FS2/GalSBI learning trajectories and learning-rate intervention

## Owner objective and boundary

Explain why FS2 learns the measured–intrinsic relation better and improve
GalSBI eventually. Keep simulation, measurement, usable flags, catalogues,
architecture, physical outputs and retained model identities fixed. No PSF
intervention in this study. Two GPUs maximum across all stages.

## Stage 1: instrumented native NLL replays

Use each original domain recorded in its base/flow/protocol.json: FS2 base
labels and GalSBI harmonized labels. The FS2 label history is not silently
replaced by harmonized65 labels. Use all160 native training and40 validation
cases, all usable rows from both legs, original stored preprocessors/config,
seed611,140epochs×4million rows, batch8192, AdamWlr1e-4, clipping5 and
ReduceLROnPlateau(factor.5,patience4,absolute threshold1e-4,minlr1e-7).
Use original stored500k validation indices for NLL and checkpoint selection.
Original historical checkpoints are evaluated as fixed reference models.
Absolute NLL across families has different normalization; compare NLL changes
within each family. Native replays may not be bitwise historical reproductions
across GPU kernels: report first-epoch discrepancies and realized schedules.

Save complete model/optimizer/scheduler states at diagnostic epochs
0,1,2,5,10,15,20,25,30,35,40,50,60,70,80,90,100,110,120,130,140, plus selected
best-NLL model/state and final state. Diagnose fixed160k samples of matched
g0 rows with intrinsic circularized Re[.15,.25) from training and validation,
uniform within that input subset, with no measured-property sample restriction.
The sample seed is202611055001(train)/202611055002(validation); save exact
pair indices and matching/dropped-row receipts. Samples are diagnostic only.

At each recorded epoch, use two16-draw antithetic banks and identical latent
streams across epochs. Bin measured and generated outputs separately in
radius0.1px bins2–5 with open tails, at mag<25 and25.8. Save per-case count,
intrinsic-shape power, shape–intrinsic dot and negative-dot count. Save observed
joint log density by the same measured bins and per-case uncut local NLL.
Also save profiles and input counts for48 strata(n4×true-mag4×|e|3), allowing
an explicitly coarse common-input-mixture comparison. Any unsupported mass
must be reported. Diagnostics do not enter checkpoint selection or gradients.
Use these traces to distinguish early failure, later deterioration, local
versus global likelihood trade-offs, and changing selected counts.

## Stage 2: causal learning-rate test on GalSBI

After the two native replays finish, fork GalSBI's complete epoch20 state.
Keep its data, batches, AdamW moments, architecture and diagnostic streams
fixed. Continue epochs21–140 using the **realized FS2 replay learning rate**
for each corresponding epoch, rather than GalSBI's plateau decisions. Use the
rate consumed by training that epoch, not the post-validation next-epoch rate.
No manual high-rate reset at the final checkpoint. The native GalSBI replay
is the control sharing the exact first20 epochs. Save both native and
intervention states so any later refinement can start from a real checkpoint.

This tests the effect of following FS2's rate schedule on GalSBI, not a unique
population-versus-observing-condition decomposition. A failure does not prove
architectural incapacity. A positive result must survive full40-case held-out
alignment/count, NLL and selected-response checks at both magnitude cuts and
neighboring radius cuts. Only then is a joint constant-shear validation
justified; no empirical calibration offsets or model-preset changes.

## Definitions and reporting

Use doc/CONVENTIONS.md sections3–8. A=sum(g_meas dot g_intrinsic)/sum(|g_intrinsic|²);
alignment error=A_flow−A_simulation, negative-dot fraction is not negative-T
fraction. Full response DeltaR=R_simulation−R_model is a separate quantity.
Case bootstrap errors condition on fitted models and omit retraining and
shared FS2 noise uncertainty. Trajectory samples and model selection make
this development evidence, not untouched acceptance.

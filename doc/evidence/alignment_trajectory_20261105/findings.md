# Learning trajectories: original FS2 and GalSBI flows

## What is established

Both native140-epoch replays reproduce historical validation NLL exactly at
every epoch. The GalSBI schedule fork holds its data, initial20 epochs, AdamW
state and batches fixed while replacing only subsequent learning rates with
those actually consumed by FS2. Fork diagnostics agree to1.35e-13.
All training/evaluation/report jobs finished successfully; see
[compact completed receipt](completed_receipt.json) and external
`sbsi_caches/alignment_trajectory_20261105/final_report.json`.

The schedule change improves GalSBI likelihood and measured–intrinsic alignment.
It does **not** produce a generally better shear-response model. The second
response component deteriorates at the main cuts. No joint constant-shear
result or accepted replacement follows from this experiment.

## Selected-model validation

All40 harmonized validation cases; original NLL-selected checkpoints versus
the new NLL-selected schedule intervention. Alignment diagnostic: intrinsic
circularized Re0.15–0.25arcsec and independently measured/generated radius
2.7–2.8pixels. These are absolute alignment errors, not percentages or shear m.

| Model | Alignment error,mag<25 | Alignment error,mag<25.8 | Common-sample GalSBI NLL |
|---|---:|---:|---:|
| FS2 original NLL | +0.03229 ±0.00463 | +0.04466 ±0.00342 | not comparable across families |
| GalSBI original NLL | +0.41323 ±0.00472 | +0.16668 ±0.00287 | −3.69413 |
| GalSBI FS2-rate schedule | +0.25333 ±0.00480 | +0.08846 ±0.00273 | −3.83296 |

Likelihood is evaluated on the same500k rows with identical target transforms.
The alignment error decreases39% atmag25 and47% at25.8. Local observed joint
NLL also improves, so this gain is not only a change in selected-bin counts.
The discrepancy remains on training data: endpoint mag25 error+0.39975 native
versus+0.24918 intervention, compared with validation+0.39796/+0.24367 on the
fixed160k diagnostic samples. These endpoint values use different rows and
checkpoints from the full selected-model table; do not mix them.

## Full response exposes a remaining problem

Absolute DeltaR=simulation−model; independently cut each shear leg and use each
branch's own baseline normalization. These are standalone flow responses,
not joint constant-shear calibration. Whole-case1sigma SEs condition on fitted
models and do not include retraining uncertainty.

| Model/cut | DeltaR11 | DeltaR22 |
|---|---:|---:|
| Original GalSBI NLL,r>3,mag<25 | −0.00943 ±0.00334 | +0.00537 ±0.00380 |
| FS2-rate GalSBI,r>3,mag<25 | −0.00497 ±0.00339 | +0.02770 ±0.00384 |
| Original GalSBI NLL,r>3,mag<25.8 | −0.00155 ±0.00298 | +0.01249 ±0.00310 |
| FS2-rate GalSBI,r>3,mag<25.8 | +0.00149 ±0.00300 | +0.02956 ±0.00312 |

R11 alone would hide the failure in R22. At the tighterr>=3.75,mag<25.8 cut,
DeltaR11 improves from−0.02921 to−0.01611 and DeltaR22 from−0.02984 to+0.00628.
The effect is selection-dependent; local alignment and global likelihood do
not suffice to select a calibrated model. Retained regularized GalSBI is a
different checkpoint (main mag25 DeltaR11+0.00695,DeltaR22+0.00730); it is
included separately in the receipt and must not be confused with NLL control.

## What this says about FS2 versus GalSBI

Atmag25, the native learning gap is already present at epoch20, before the
first GalSBI rate reduction at27. Earlier rate reduction then limits later
catch-up: the exact-state fork establishes a causal contribution for this
seed and dataset, but cannot explain the whole gap. Atmag25.8 the early
ordering reverses, so there is no universal early FS2 advantage across cuts.

Coarsely matching the input mixture across48 Sersic-index/true-magnitude/
intrinsic-ellipticity cells leaves endpoint mag25 errors+0.05729 FS2 versus
+0.40399 GalSBI, with zero unsupported input mass. Atmag25.8 they are+0.06227
versus+0.12304. This excludes a simple explanation solely from proportions
in those coarse cells. It does not match exact contexts, physical band/SED,
measurement noise, or sub-bin populations. A unique population-versus-band
explanation remains unresolved. Train/validation agreement supports underfitting
on the represented population; it does not prove insufficient architecture.

![Learning trajectories](final_trajectories.png)

## Next controlled test

The [full-budget frame study](../alignment_frame_full_20261105/rule.md#motivation-and-controlled-difference)
changes only output-shape representation relative to the completed FS2-rate
Cartesian control. Same160 cases, original standardizers/configuration, seed,
epoch samples, optimizer and140epoch budget. It removes the budget/scaling
confounds in the earlier short frame pilot. Full-matrix and neighboring-cut
validation are mandatory; a local alignment gain alone is not acceptance.

## Reproducibility and conventions

See [predeclared design](rule.md#stage-1-instrumented-native-nll-replays),
[selection conventions](../../CONVENTIONS.md#3-population-and-selection-order),
[shape conventions](../../CONVENTIONS.md#4-shapes-and-shear),
[response conventions](../../CONVENTIONS.md#6-response-estimators) and
[seeds/model selection](../../CONVENTIONS.md#8-seeds-and-model-selection).
A=sum(g_measured dot g_intrinsic)/sum(|g_intrinsic|²); error=A_flow−A_sim.
Negative-dot fraction is not negative-T fraction. FS2 native training used its
original base labels, GalSBI its harmonized labels; final comparison uses
harmonized validation labels for both. Coarse mixture comparison uses native
endpoint diagnostics. Case errors omit FS2 shared-noise and retraining
uncertainty. Development data and inspected selection rules preclude claiming
independent acceptance. No retained weights/presets or measurement code changed.

Jobs17758686(native),17758694(fork),17758743(selected evaluation),
17758753(native report),17758754(final report) all exit0. Cancelled redundant
pending status17758885 after this immediate completed-results audit. Verified
140epoch exact historical NLL replay, exact model reload, measured evaluation
replay, fork profile consistency, small reducer/bin tests and inspected the
final trajectory figure. Full raw outputs remain external.

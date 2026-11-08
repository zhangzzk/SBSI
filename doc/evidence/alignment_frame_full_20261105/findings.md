# Full-budget intrinsic-frame GalSBI test

## Completed result

Changing only output-shape representation relative to the matched Cartesian
FS2-rate control improves held-out joint likelihood and the original low-radius
alignment discrepancy. It worsens alignment immediately above the selection
cut and does not uniformly improve the full shear response. This is useful
evidence about learning geometry, not a calibrated-model replacement.

Training17762865, evaluation17762866 and report17762867 all completed exit0.
Best likelihood checkpoint epoch136; exact saved-model reload. Both arms have
original160 cases, identical base initialization, original standardizers,
140epochs×4m, batch8192, same optimizer and realized FS2 learning rates.
Only output coordinates differ. See [predeclared rule](rule.md#motivation-and-controlled-difference)
and [completed receipt](completed_receipt.json). Large raw results remain at
`sbsi_caches/alignment_frame_full_20261105/comparison.json`.

## Alignment and density

All40 harmonized validation cases. Intrinsic circularizedRe0.15–0.25arcsec;
independently measured/generated selection. Error=A_flow−A_sim where
A=sum(g_measured dot g_intrinsic)/sum(|g_intrinsic|²). These are absolute errors,
not percentages or constant-shear m. Uncertainties are fixed-model whole-case1sigma.

| Model | Radius2.7–2.8,mag25 | Radius3.0–3.1,mag25 | Radius2.7–2.8,mag25.8 | Common500k NLL |
|---|---:|---:|---:|---:|
| Original GalSBI NLL | +0.41323 ±0.00472 | +0.09632 ±0.00419 | +0.16668 ±0.00287 | −3.69413 |
| Matched Cartesian,FS2-rate | +0.25333 ±0.00480 | +0.03438 ±0.00408 | +0.08846 ±0.00273 | −3.83296 |
| Matched intrinsic frame | +0.09174 ±0.00485 | +0.11492 ±0.00408 | +0.03505 ±0.00277 | −3.88700 |

The frame-minus-Cartesian change is−0.16159 ±0.00149 in the firstmag25 bin,
but+0.08054 ±0.00087 just above the cut. Atmag25.8, the corresponding changes
are−0.05341 ±0.00071 and+0.05638 ±0.00067. These paired-case errors preserve
model covariance but omit training variability; do not interpret them as
complete uncertainty on architecture superiority.

At radius2.7–2.8,mag25, simulation alignment is−0.34055. Original NLL flow
predicts+0.07268, schedule control−0.08722, and frame−0.24881. The new model
recovers much more of the negative alignment, with52985 measured selected
objects versus52722 mean generated selected objects. At3.0–3.1, simulation
alignment is+0.39823 versus Cartesian+0.43261/frame+0.51315. Counts remain close
(114448 measured,114662 Cartesian,113063 frame), despite wrong shape moments.
The full profile confirms a redistribution of error across the cut; it is
not evidence that the low-radius alignment alone determines calibration.

![Matched density and alignment comparison](flow_comparison.png)

## Full response and decision

Absolute DeltaR=simulation−model, forward matched-leg response, independently
cut each leg and use each branch's own baseline normalization. These are
standalone flow checks on held-out flow data, not joint constant-shear results.

| Cut/model | DeltaR11 | DeltaR22 |
|---|---:|---:|
| r>3,mag25,Cartesian | −0.00497 ±0.00339 | +0.02770 ±0.00384 |
| r>3,mag25,frame | −0.01741 ±0.00337 | −0.00705 ±0.00382 |
| r>3,mag25.8,Cartesian | +0.00149 ±0.00300 | +0.02956 ±0.00312 |
| r>3,mag25.8,frame | −0.00906 ±0.00302 | +0.00083 ±0.00312 |
| r>=3.75,mag25.8,Cartesian | −0.01611 ±0.00471 | +0.00628 ±0.00351 |
| r>=3.75,mag25.8,frame | −0.01319 ±0.00479 | −0.00093 ±0.00351 |

The large R22 failure of the schedule control improves, but main-cut R11
worsens. The R11 deterioration is concentrated in the same intrinsically
small objects: Re0.15–0.25arcsec contributes−0.01016 of the total−0.01244
frame-minus-Cartesian change at r>3,mag25 (about82%). This is an additive
contribution to the full-sample response, not that bin's separately normalized
bias. It connects the remaining response problem to the studied population,
but does not prove that the alignment statistic alone causes it. Off-diagonalR12 remains near−0.010 at both main cuts; full matrices
and paired changes are preserved externally. Do not accept the model based
only on density or the original alignment bin. No joint constant-shear run
or model-preset change was made.

## Interpretation and next experiment

We can substantially improve the GalSBI relation while keeping the GalSBI
population/images/measurements fixed. Optimization schedule and coordinate
representation both matter for this seed. This does not isolate which
population/band/measurement differences make GalSBI harder than FS2, nor show
that the remaining alignment error entirely explains constant-shear bias.

The remaining profile suggests testing explicit conditioning of shape on
measured/generated radius and flux: p(radius,flux|input) times
p(shape|radius,flux,input). This is a hypothesis about learning the dependence,
not a correction pasted onto the output. The
[factorized study](../alignment_factorized_20261105/rule.md#evidence-and-hypothesis)
tests Cartesian and intrinsic-frame forms with matched data/budget and nearly
the same parameter count. Physical transforms/density Jacobians remain exact.

## Verification and limitations

Coordinate inverse/norm, determinant1 with unequal scales, physical support,
density/Jacobian discrepancy5.08e-7, identical Cartesian wrapper, exact saved
model reload, original measured replay and paired response-reducer consistency
all pass. Sources parse and shell launchers pass syntax checks. Completed
plot17764011 was inspected; corrected common-sample NLL reference line was
regenerated by17764103. Cancelled redundant pending check17762868 after this
immediate audit. Case errors omit retraining uncertainty; repeated use of these
validation scenes makes this development evidence. No independent acceptance.
Conventions: [selection](../../CONVENTIONS.md#3-population-and-selection-order),
[shapes](../../CONVENTIONS.md#4-shapes-and-shear),
[responses](../../CONVENTIONS.md#6-response-estimators).

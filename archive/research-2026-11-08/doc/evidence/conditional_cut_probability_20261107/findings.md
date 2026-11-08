# Conditional cut-probability experiment

## Completed comparison: a modest improvement across the nine cuts

Training/pilot17817062, all three persistent workers17817064 and report17817065
completed exit0:0. Both models are the predeclared step64 endpoints, with no
outcome-selected weight or checkpoint. All eight field/axis pairs pass the
original replay gates. The new conditional probability objective improves the
held-out hard-cut score and the pooled absolute coherent bias at all nine cuts
on both axes. This is a useful development result, not a solution of the
remaining bias or independent calibration acceptance.

The central cut is measured/generated FLUX_RADIUS >3 pixels and MAG_AUTO <25,
as appropriate on each side, with predicted usability and independent per-leg
generated selection on the model side. Four existing GalSBI constant-shear
fields, h=0.02, 64 portable paired draws, fixed models/measurements. Bias is
m=100(R_sim/R_model-1); errors below are paired delete-one-field standard errors
in percentage points. The original source manifest retains the seeds and IDs.

| Quantity | Matched control | Added conditional probability loss | Paired change |
|---|---:|---:|---:|
| Central m1 (%) | +0.7410 | +0.6377 | -0.1033 ± 0.0084 pp |
| Central m2 (%) | +1.1172 | +1.0025 | -0.1146 ± 0.0140 pp |
| Nine-cut RMS m1 (%) | 1.0501 | 0.9440 | -0.1060 ± 0.0534 pp |
| Nine-cut RMS m2 (%) | 0.9860 | 0.8677 | -0.1183 ± 0.0121 pp |

The new endpoint's absolute central SEs are 0.7165/0.4847 percentage points,
much larger than the uncertainty on the paired change. The cut results are
correlated. All nine pooled absolute biases also improve with actual usability.
The signed central residual decreases in every field, but absolute nine-cut
RMS worsens in case1000 on both axes; its error is already negative over much
of the cut range. The other three fields improve in RMS on both axes. See
[all cuts and field table](cut_results.md), [case audit](case_audit.json) and
[complete original report](report.json).

## What changed in probability and shapes

On the original 500k validation rows across forty held-out flow-validation
cases, the mean hard-cut probability score decreases from0.07168472 to0.07154394,
a paired change of -0.00014078 ±0.00000887. The same direction holds on training
rows, the soft training target, and the predeclared0.025/0.1 smoothing checks.
Validation photometry NLL changes +0.000035 ±0.000185: no detected deterioration
at this precision. Conditional shape parameters and all other nonphotometry
tensors remain exactly unchanged; direct conditional density at fixed inputs
is exact. Subtracting rounded joint/photometry log densities in the aggregate
produces only a1.5e-10 difference. These checks certify the fixed factor, not
unchanged selected-shape behavior. [Probability and factor diagnostics](probability_report.json).

The selected-neighbour score error falls from0.003367 to0.001894 on unsheared
validation rows, a smaller improvement than the old static-neighbour loss
achieved. Nevertheless, the complete shear response improves here whereas
that earlier loss worsened it. The score alone therefore remains inadequate
as a calibration criterion.

At the central cut with predicted usability, the flow response decreases by
0.000792/0.000697 in R11/R22, while the emulator contribution increases by
0.001714/0.001718. The net model response increases by0.000923/0.001020,
reducing the positive simulation-minus-model residual. These are changes in
model contributions under different generated photometry/membership, not
measurements of separate physical component errors. Freezing shape weights
did not freeze the selected flow response. [Full component table](results.md).

## Completed numerical audit and next decision

The repeated control has all111 state tensors bitwise equal to the earlier
frozen-shape control. Simulation sums, selected counts and score association
replay exactly. An initially overstrict cross-run raw-sum equality check finds
floating-point differences from pooled emulator arithmetic and hardware;
the existing response replay criterion is unchanged, with largest flow-response
difference1.20e-13. The independent small-sum reduction reproduces all216
pooled contrast values and their jackknife SEs, plus each endpoint's nine-cut
RMS and its SE. [Audit source](audit_cases.py), [completed receipt](completed_audit.json).

This bounded experiment is complete. Preserve both endpoints. A sensible next
test is replication of the same fixed pair on more existing scenes before
changing the objective, loss strength or training duration. These four fields
do not show that the remaining approximately percent-level residual is fixed,
and no larger evaluation or new training is launched by this completion audit.
The earlier broad array17788286 remains held; automated follow-ups remain off.

## Startup checks and current status

The following is the preserved startup record, superseded by the completed
comparison above.

Training allocation 17817062 has started on an A40-16Q and passed the analytical
and numerical gates. Its auxiliary loss gradient agrees with direct autograd
to relative error 6.30e-8, and the unchanged response gradient check gives
8.41e-8. All 61 nonphotometry state tensors are unchanged at initialization;
the direct conditional shape density matches exactly on 1,024 measured inputs.
The control continuation has reached saved optimizer checkpoints.

The four training-only calibration batches set the extra probability-loss
weight to 220.3031483586. Baseline photometry gradient norms are 39.87, 46.66,
45.17, 58.78; unit-weight probability gradients are 0.06591, 0.04328, 0.05607,
0.05065. Their frozen RMS ratio gives an initial auxiliary gradient magnitude
of 25% of the baseline. This coefficient scales a training loss; it is not an
offset or correction to a shear estimate. See [calibration receipt](weight_calibration.json).
No validation or constant-shear value entered the calculation.

The [protocol](rule.md#authorized-comparison-and-scope) keeps both arms at
64 steps. The weighted arm, held-out probability/shape diagnostics, coherent
first-field gate, seven remaining field/axis tasks and paired report are still
pending at this startup record. No scientific improvement is established.
See [job identifiers](jobs.json). Dependent workers 17817064 and report17817065
remain gated; broad17788286 is held and automated follow-ups are disabled.

## What will answer the question

The main comparison is weighted versus the matched photometry-only control.
Check whether conditional hard-cut scores improve on disjoint validation
scenes, how the selected shape moments change, and whether the complete
coherent m improves across the nine target cuts on both axes. Read the exact
cut and uncertainty definitions in [the protocol](rule.md#correctness-gates-and-behavioral-checks).
The static neighbour score and single-leg shape errors are diagnostics; neither
is the final shear bias. The original NLL and nine-cut models remain historical
references, not alternate selected endpoints.

The [analytical fixtures](verification.json) check independent-bank expectation
and gradient, detect the variance bias of reusing a bank, and distinguish a
conditional score from a squared population mean. The [reducer fixtures](reducer_verification.json)
check unequal case sizes, paired jackknife, selected-shape normalization and
unchanged conditional density. Synthetic fixture scores printed in the startup
log are test values, not fitted-model results.

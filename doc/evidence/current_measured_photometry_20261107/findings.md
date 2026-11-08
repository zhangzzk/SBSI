# Current model: residual remains with measured photometry and membership

> 2026-11-08 update: the numbers below retain the original unpaired shape
> measurements. The [paired-seed calibration](../selected_seed_calibration_20261108/findings.md#completed-result)
> supersedes their absolute measured-bias interpretation. Frozen model shifts
> remain valid. Old measured-input/oracle sums require matched re-evaluation
> before comparison with new shape validity and measurement outcomes.

## Completed result and decision

All40 scenes/both axes and the report completed successfully. Supplying actual
size, flux, per-leg usability and membership leaves central biases
m1=+0.11693±0.31325% and m2=−0.95552±0.24107%. The second component is negative
at every target cut. Photometry-only improvement is therefore not a complete
remedy within the current conditional-shape/emulator composition.

The two latest photometry endpoints give the same measured-input prediction:
all61 nonphotometry tensors, conditioning states, pilot full-field predictions
and subsequent first-block conditional draws match exactly. The earlier
measured-input results used different checkpoints; this closes that gap.

## Matched central-cut comparison

All40 scenes; actual usability independently per leg in every row below.
Radius >3 pixels, magnitude <25. Errors are whole-scene jackknife SE; bias
values are percent and uncertainties are percentage points.

| Model/input choice | m1 (%) | m2 (%) | Half component difference (pp) |
|---|---:|---:|---:|
| Control: generated photometry/selection | +1.12408 ±0.31150 | -0.04535 ±0.24762 | +0.58472 ±0.19844 |
| Extra loss: generated photometry/selection | +1.00553 ±0.31156 | -0.15975 ±0.24745 | +0.58264 ±0.19831 |
| Both: actual photometry/selection | +0.11693 ±0.31325 | -0.95552 ±0.24107 | +0.53622 ±0.19925 |

Relative to the extra-loss endpoint, measured inputs change m1 by
−0.88860±0.03704pp and m2 by−0.79577±0.02703pp. The half component difference
falls by0.04642±0.01538pp, but remains+0.53622±0.19925pp. This is a joint input
substitution, not an additive causal allocation of the original error.

For comparison, the normal predicted-usability extra-loss result was
+0.80810%/−0.33352%. Do not mix that row with the actual-usability comparison
when assigning a photometry/selection effect.

Conventions: m=100(Rsim/Rmodel−1); antithetic±0.02; pool scene sums before
ratios; independent per-leg normalization and output cuts. [Bias definition](../../CONVENTIONS.md#7-multiplicative-bias),
[response conventions](../../CONVENTIONS.md#6-response-estimators),
[protocol](rule.md#population-identity-and-numerical-gates).

## Scene and cut checks

The original4 measured-input biases were−0.1086%/+0.3672%, whereas the
additional36 give+0.1420%/−1.1022%. Both cohorts are reported separately;
the original4 cannot establish the broader conclusion. On all40 scenes, the
nine-cut m2 values range−0.9939% to−0.5751%; m1 ranges−0.0227% to+0.2250%.
These cuts share objects and are not nine independent tests.

Joint nine-cut RMS becomes0.58681%. Against the extra-loss model with actual
usability/generated selection it changes by−0.15409±0.24471pp; against its
normal predicted-usability result it changes by−0.01842±0.21708pp. Neither RMS
improvement is resolved at this scene precision. Thus the oracle input
substitution cannot be described as a clear general calibration repair.

The two independent32-draw banks differ by+0.00888±0.01076pp in m1 and
+0.00032±0.01388pp in m2 at the central cut. This sensitivity check does not
suggest that the near−1% second-component residual comes from these finite
flow draws. It is not a full integration-error or fitter-repeat study.
[All target cuts](all_cuts.md), [full cohort/matrix report](report.json).

## What comparison with earlier models establishes

Under the same measured inputs and common draws, the current conditional
model differs from the original nine-weak model by+0.11958±0.00117pp in m1
and−0.22246±0.00124pp in m2. The half component difference grows
+0.17102±0.00083pp. The underlying model response changes are:

| Current minus original nine-weak | R11 | R22 |
|---|---:|---:|
| Flow contribution | −0.000617408 | +0.001563013 |
| Emulator contribution | −0.000466066 | +0.000467672 |

The emulator is unchanged and receives the same measured photometry. Its
contribution can nevertheless change because disk transport acts on different
base shapes. This explains why a fixed emulator is not a fixed response term.
These are model changes, not independent measurements of either physical error.

Crucially, these checkpoints belong to different training branches. The current
shape factor descends from neighbour-NLL checkpoint12ca7856 through the
neighbour-moment **control**6fc0c373. Original nine-weak888fb044 is not its direct
starting checkpoint. The comparison does not attribute the difference to the
latest probability loss, which left shapes exactly unchanged, or isolate the
neighbour feature from training history. [Audit](completed_audit.json).

## Next discriminating check

Stop treating the conditional shape factor as known-correct. Before another
loss, test its predictions directly against measured shapes at measured
photometry in the existing zero-shear validation images. Those images permit
checking conditional shape means, variances and directional second moments
without assuming that the coherent residual belongs entirely to the flow.
The [bounded zero-shear check](../conditional_shape_check_20261107/rule.md#question-and-scope)
uses four pinned earlier/current models and common draws; it does not retrain.

A detected error would support a conditional-shape learning hypothesis. A null
result would leave shear-derivative and neighbour-composition errors open.
Neither identifies the full primary/neighbour physical budget. The earlier
small physical replay remains too noisy for that purpose. No empirical offset,
shape rescaling, checkpoint promotion or new acceptance claim follows here.

## Validation and limitations

Pilot17821126, workers17821114_0–5 and report17821115 all complete0:0. Pilot
original-model full-field response replay is exact. All80 new case/axis outputs
pass measured-photometry, selection-mass, simulation, feature, conditional-draw
and half-bank checks. The first startup17821113 failed before prediction on
NumPy JSON serialization; the exact recursive-state comparison fixed it.

Independent direct-sum audit reproduces15,330 point/paired/jackknife scalar
statistics to maximum difference8.89e-14, verifies240 current/generated file
hashes and checkpoint identities, and adds RMS contrasts and separately labelled
historical comparisons. Its initial antisymmetric-matrix convention was reversed;
aligning that diagnostic with the report convention closed the check without
changing any evaluated response or reported science result.

Across40 scenes×2 axes×2 legs, zero crossmatch rows lack a shape;228,683 shape
rows without crossmatches are excluded.94,712,853 parent rows lack crossmatches
and are not counted as usable. All counts are deduplicated across model arms.
The complete matching totals remain in the audit/receipts.

These are reused development scenes with fixed fitted models and measurement
realizations. Scene errors do not include training-seed uncertainty or an
independent repeated-measurement experiment. Broad17788286 remains held and
automated follow-ups remain disabled. Production presets are unchanged.

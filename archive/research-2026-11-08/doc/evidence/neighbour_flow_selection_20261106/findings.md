# The selection association is underfit on the flow's own data

## Main finding

This failure is present on training and held-out validation data, before applying
the flow to coherent constant-shear scenes. At radius>3px/magnitude<25, generated
selection gives too little weight to objects with stronger neighbour coupling.
The added scalar neighbour feature improves the association modestly; the
existing response regularizer worsens it again. A population-transfer difference
cannot be the sole explanation for this particular association error.

For each object, B_ref is the frozen emulator's summed neighbour coefficient
at reference radius3.5px/magnitude24.5. The diagnostic is

    D_B = mean(B_ref | measured selection)
          - sum[P_flow(selection | context) * B_ref] / sum[P_flow(selection | context)].

This is a selected-neighbour coefficient diagnostic, not a shear-response error
or a multiplicative bias. All rows are usable in their original simulation leg.
Below, g0 means the original unsheared leg; uncertainty is whole-case bootstrap
SE on fixed sampled rows and fitted models.

| Flow | Training g0 D_B | Validation g0 D_B |
|---|---:|---:|
| Original Cartesian NLL | +0.003320 ± 0.000416 | +0.003117 ± 0.000513 |
| Matched continuation, no new input | +0.002994 ± 0.000416 | +0.002780 ± 0.000512 |
| Added neighbour input, NLL | +0.002678 ± 0.000414 | +0.002520 ± 0.000513 |
| Added neighbour input, nine-cut | +0.003322 ± 0.000414 | +0.003148 ± 0.000511 |

The feature improves D_B by0.000316 on training g0 and0.000261 on validation g0
relative to the matched control. Subsequent regularization increases it by
0.000644 and0.000628 respectively. Thus its adverse change is about twice the
gain from adding the feature in this experiment.

## Which objects and likelihood factor

Using training-derived score quintiles, the neighbour-conditioned NLL flow
underpredicts the selected fraction in the highest-coupling fifth by0.94pp on
training rows and0.93pp on validation rows (pooled g0/g05). It overpredicts
selection in the lowest-coupling fifth by0.34pp and0.30pp. After regularization,
the high-coupling deficits grow to1.12pp and1.11pp. Opposing selection-rate
errors partly cancel in the overall count, while the weighted association stays
wrong. These quintile rates are diagnostics; they are not new rejection rules.

The new feature is used: replacing its input with the training mean worsens
validation NLL by0.009333 per row and increases pooled D_B by0.000325. Adding
the feature improves both photometry NLL (-0.001885 relative to the matched
NLL control) and conditional-shape NLL (-0.001783). Better total likelihood
therefore does not imply recovery of this selected moment.

For the augmented model, adding the nine-cut response penalty worsens validation
photometry NLL by0.005461 and conditional-shape NLL by0.001198. The penalty
constrains all four response-matrix components, but it does not explicitly
constrain the selected-neighbour moment above. This is evidence of a tradeoff
in the learned distribution, not proof that a single moment accounts for every
constant-shear residual.

## What this suggests testing next

Keep the response constraints at the owner's nine target cuts and test an
additional training-data constraint on selection weighted by neighbour coupling,
alongside selection mass. Estimate targets exclusively from the original
training measurements and evaluate on held-out rows and the frozen constant
scenes. A matched control is needed to distinguish that constraint from more
training. Matching D_B alone would not establish joint calibration; the full
emulator depends on actual photometry and the transported shape distribution.
No such new training or empirical correction was applied in this diagnostic.

## Validation and execution

One persistent GPU job17809769 completed in3m33s, loading six distinct frozen
models once and evaluating eight arms on500,000 training and500,000 validation
rows. It preserved both simulation legs, case identities and64 shared portable
antithetic draws. All photometry-only draws agree exactly with the corresponding
full joint draws. Cached input-score alignment is exact, full/half-bank moments
agree, and saved validation NLLs replay within2.8e-7. The two32-draw banks differ
by at most0.000102 in the displayed training g0 D_B values, below their case SEs.

See [all results](results.md), [full numerical report](report.json),
[protocol](rule.md#question-and-frozen-endpoints), and the
[completed forty-scene axis audit](completed_audit.json).

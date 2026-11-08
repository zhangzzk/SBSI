# Completed paired-seed measurement, 2026-11-08

## Full40 result

All2,667 chunks completed. Of5,412,739 originally common-selected IDs,
5,411,332 remain valid in all four new fits; exclude1,407 from both sides.
The old and new rows therefore use exactly the same galaxies.

| Quantity | Original fits | Paired-seed fits |
|---|---:|---:|
| R11 | 0.898733 | 0.897227 ±0.001164 |
| R22 | 0.890108 | 0.895106 ±0.001397 |
| R11−R22 | 0.008625 ±0.003183 | 0.002121 ±0.001439 |
| R12−R21 | −0.004661 ±0.003260 | 0.000119 ±0.001671 |

Errors are whole-scene jackknife standard errors across40 scenes, with
pooled sums normalized separately per leg before differencing ±0.02 shear.
Rows of R are shape components; columns are shear components. Definitions:
[conventions](../../CONVENTIONS.md#6-response-estimators).
The paired change in diagonal gap is−0.006504 ±0.003283, approximately1.98SE;
the new gap is1.47SE from zero. This supports measurement sensitivity and
reduced stochastic fitting noise; it neither enforces symmetry nor proves
that all calibration bias is removed. Hardware-dependent numerical fitting
variation remains a limitation. No model has been retrained.

## Validation and next comparison

Report17821860 completed0:0. The completed audit checks all2,667 source/hash/
repeat receipts and independently reproduces36 report statistics from saved
sums; see `completed_audit_20261108.json` and `report.json`. This supersedes
the partial20/22-scene estimates, whose populations differ.

This fixed common-galaxy population is unsuitable for direct division by the
independently selected model response. Continue with the
[per-leg calibration protocol](../selected_seed_calibration_20261108/rule.md),
which reuses these fits and includes additional eligible galaxies and recovered
old-invalid fits. Reused development scenes do not provide independent acceptance.

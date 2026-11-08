# Matched architecture controls: constant shear

## Scope and status

[Protocol](rule.md#question-and-predeclared-scope) freezes the full-dimensional
Cartesian and intrinsic-frame flows trained with the same FS2 learning-rate
schedule and140epoch budget as the factorized candidates. All40 constant scenes,
original emulator/C8,64 antithetic draws and all cuts stay fixed. Array17771754
initially ran four workers alongside two conditional-photometry workers. Both
conditional jobs completed; the control array now permits six one-GPU workers. Pilot17771755 follows the first four scenes; full17771756
follows all80 evaluations. Status17771762 records a bounded check after about2h.

The [completed factorized comparison](../alignment_factorized_constant_20261105/findings.md#completed40-case-comparison)
shows improvement at radius>3 and deterioration at radius>=3.75. These controls
separate the architecture/allocation change from training-schedule and penalty
changes relative to the retained model. At launch no result was available for these
matched controls. No production model or simulation configuration changes.

## Resource continuation

Owner authorized seed replication; controls now throttle4, with pending tasks
on inter, alongside at most2 A40 training/standalone tasks. Scientific settings
and full-report dependencies remain unchanged. Combined six-GPU ceiling holds.


## Completed40-case matched controls

All80 evaluations and full reducer17771756 completed successfully. Main
selection: predicted usability, independent generated radius>0.60 arcsec and
mag<25, no truth-property cut. m=100(Rsim11/Rmodel11-1), case-bootstrap SEs
conditional on fitted models/catalogues, in percentage points.

| Flow | m (%) |
|---|---:|
| retained | +2.4288 ± 0.3102 |
| factorized_cartesian | -0.2739 ± 0.2995 |
| factorized_frame | +0.4373 ± 0.3042 |
| joint_cartesian | +0.9040 ± 0.3029 |
| joint_frame | -0.7053 ± 0.3010 |

Factorized minus full joint Cartesian m difference−1.17794±0.00885pp;
matched intrinsic-frame difference+1.14257±0.01466pp. These paired SEs cancel
shared simulation noise; they are not absolute calibration uncertainties.
The Cartesian factorization benefit remains when the140epoch FS2-rate recipe
and NLL checkpoint criterion are matched. The comparison with retained flow
also changes training schedule/budget/selection/regularization, so its entire
improvement cannot be attributed to factorization or learning rate alone.
Coordinate choice affects the result; no architecture is uniformly best over
all cuts. [full_report.json](full_report.json).

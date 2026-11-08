# Neighbour-conditioned flow outcome

## Four-scene matched comparison

All four frozen endpoints complete on cases1000/1040/1080/1120 and both shear
axes. Use the original classifier/emulator and generated photometry/selection.
The table below uses actual per-leg usability, radius>3px and magnitude<25.
Values are m in percent with paired whole-case bootstrap SE in percentage
points; these four reused scenes are development evidence.

| Endpoint | m1 (%) | m2 (%) |
|---|---:|---:|
| Matched control, NLL | -0.2614 ± 0.5990 | +0.4870 ± 0.4099 |
| Added neighbour input, NLL | -0.2840 ± 0.5972 | +0.3901 ± 0.4092 |
| Matched control, nine-cut | +1.0507 ± 0.6082 | +1.4466 ± 0.4201 |
| Added neighbour input, nine-cut | +1.0564 ± 0.5981 | +1.3794 ± 0.4132 |

The feature's paired change is small: -0.0226/-0.0969pp for NLL on axes1/2,
and+0.0057/-0.0673pp after nine-cut regularization. The strong regularization
shift remains. [Full report](report.json) contains generated/predicted-usability
results, true-selection diagnostics, all nine cuts and paired components.

## Selection association and likelihood

The selected-neighbour reference-score error changes from0.002785 to0.002484
for NLL (about11% smaller), and0.003375 to0.003112 after regularization (about8%
smaller). The overall selection fractions remain close to measured. This new
feature alone does not recover the earlier diagnostic selection predictor's
near-zero score error.

Held-out NLL does improve before response regularization: -3.921348 with the
new input versus-3.917680 for the matched control. After regularization the
values are-3.914690 versus-3.914985, respectively. Improvement in the total
likelihood therefore does not establish recovery of the particular selection
association or joint calibration.

The response penalty explicitly constrains all four response-matrix components;
R22 was not omitted. Original training/probe values are not constant-shear
component errors. No checkpoint was chosen by the reported responses and no
production model is replaced.

## Follow-up

The [own-data selection diagnostic](../neighbour_flow_selection_20261106/rule.md#question-and-frozen-endpoints)
checks whether the residual association is present in training/validation rows
and separates likelihood changes in photometry from those in conditional shape.
It holds the six unique fitted models fixed and uses one persistent GPU job.
That diagnostic has now completed: it finds the same selection association
error on training and validation rows, with regularization worsening it.
See [the follow-up findings](../neighbour_flow_selection_20261106/findings.md#main-finding).

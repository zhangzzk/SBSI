# Frozen cut-probability replication: repeatable shift, mixed calibration

> 2026-11-08 update: the numbers below retain the original unpaired shape
> measurements. The [paired-seed calibration](../selected_seed_calibration_20261108/findings.md#completed-result)
> supersedes their absolute measured-bias interpretation. Frozen model shifts
> remain valid. Old measured-input/oracle sums require matched re-evaluation
> before comparison with new shape validity and measurement outcomes.

## Completed result

All 36 additional scenes, both shear axes, and the final report completed.
The fixed conditional cut-probability loss shifts central m downward in every
additional scene on both axes. The first component moves closer to zero; the
second component's pooled point estimate moves farther from zero. The original
four-scene improvement therefore does not establish a general two-component
calibration improvement.

At measured radius >3 pixels and magnitude <25, predicted usability and
generated selection:

| Cohort | Control m1 (%) | New m1 (%) | Paired Δm1 (pp) | Control m2 (%) | New m2 (%) | Paired Δm2 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Original 4 | +0.7410 ± 0.7229 | +0.6377 ± 0.7165 | -0.1033 ± 0.0084 | +1.1172 ± 0.4869 | +1.0025 ± 0.4847 | -0.1146 ± 0.0140 |
| Additional 36 | +0.9436 ± 0.3377 | +0.8271 ± 0.3378 | -0.1165 ± 0.0035 | -0.3689 ± 0.2602 | -0.4817 ± 0.2598 | -0.1128 ± 0.0033 |
| All 40 | +0.9233 ± 0.3103 | +0.8081 ± 0.3102 | -0.1152 ± 0.0033 | -0.2205 ± 0.2482 | -0.3335 ± 0.2478 | -0.1130 ± 0.0032 |

Errors are whole-scene delete-one jackknife SE, in percentage points. Absolute
bias errors are much larger than the paired errors: a repeatable model change
does not determine the unknown population bias precisely. All comparisons use
m=100(Rsim/Rmodel−1), antithetic ±0.02 differences, independent per-leg
normalization and cuts, the fixed common draws and the same simulated images.
See [response conventions](../../CONVENTIONS.md#6-response-estimators),
[bias definition](../../CONVENTIONS.md#7-multiplicative-bias) and the
[study population/seed contract](rule.md#population-and-numerical-contract).

## Robustness across the nine target cuts

All radius >2.8/3.0/3.2 pixel × magnitude <24.9/25.0/25.1 cuts are retained.
The RMS below is the RMS of the nine pooled biases, not a mean of individual
scene RMS values. Its change can be nonlinear and uncertain near zero.

| Cohort | Component | Control RMS (%) | New RMS (%) | Paired change ± jackknife SE (pp) | Bootstrap SE (pp) | Cuts with smaller absolute bias |
|---|---|---:|---:|---:|---:|---:|
| Additional 36 | 1 | 0.9603 | 0.8314 | -0.1289 ± 0.0033 | 0.0102 | 9/9 |
| Additional 36 | 2 | 0.2266 | 0.3003 | +0.0737 ± 0.0834 | 0.0734 | 3/9 |
| All 40 | 1 | 0.9590 | 0.8306 | -0.1285 ± 0.0028 | 0.0071 | 9/9 |
| All 40 | 2 | 0.1725 | 0.2068 | +0.0343 ± 0.1450 | 0.0820 | 3/9 |

In the additional 36 scenes, individual nine-cut RMS improves in 28/36 for
component 1 and 18/36 for component 2. The central changes range from −0.1651
to −0.0795 pp and −0.1473 to −0.0751 pp respectively. Thus the direction of
the intervention replicates much better than its benefit in absolute bias.
The apparent component-2 RMS worsening is not statistically resolved by these
scene errors. [Every target cut](all_cuts.md#additional_thirty_six),
[per-scene audit](completed_audit.json), [full report](report.json).

Using actual usability while retaining generated selection does not rescue a
clear two-component claim: new central biases in the additional 36 are
+1.0201% and −0.3075%, versus +1.1409% and −0.1931% for control. Component-2
nine-cut RMS changes by −0.0160 ±0.1519 pp (jackknife). Actual-membership
results still use generated photometry internally; they are not the earlier
measured-photometry substitution and cannot be used to restate its claim.

## What changes in the response

The shape-network weights remain fixed, but the selected shape response does
change. On the additional 36 scenes at the central cut:

| Change in diagonal model response | Component 1 | Component 2 |
|---|---:|---:|
| Flow term | −0.000672879 | −0.000687242 |
| Emulator term | +0.001711357 | +0.001706888 |
| Total | +0.001038478 | +0.001019646 |

These are changes in model predictions under the training intervention, not
attributions of the physical error to the flow or emulator. Their opposite
signs explain why preserving the conditional shape network does not preserve
the full response: size/flux affect selected objects and emulator inputs.
The held-out probability-score gain from the original experiment remains a
valid result; the larger calibration check limits what can be inferred from it.

The separate [four-way experiment](../photometry_learning_information_20261107/findings.md#completed-comparison-better-probabilities-no-shear-calibration-improvement)
starts from original checkpoint888fb044, whereas this pair starts from the
already neighbour-trained6fc0c373 checkpoint. Its opposite bias shift is not a
matched contradiction and must not be pooled with this replication. Together
with its [response accounting](../photometry_response_localization_20261107/findings.md#main-finding),
these results motivate testing how selection and shape response change together,
including the selected mass used to normalize the mean. Improving passing
probability alone is insufficient evidence for improved shear calibration.

## Validation and completed jobs

Pilot17817800, all six packed workers17817802_0–5, and report17817803 completed
with exit0:0. The pilot exactly replays the original case1000/axis1 response
(maximum recorded difference0), then passes the first new scene before workers
start. The 72 additional scene/axis tasks are complete, with both fixed models.
The reducer also reproduces the original four-scene results.

The independent saved-sum audit verifies all160 result files, checkpoint and
source hashes, fixed classifier/emulator identities, and reproduces20,268
point/paired/jackknife scalar statistics to maximum absolute difference
6.58e-14. It directly pools sums and deletes scenes without importing the report
estimator. Command:

```
OPENBLAS_NUM_THREADS=1 /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python doc/evidence/conditional_cut_replication_20261107/audit_completed.py
```

Across40 scenes ×2 axes ×2 legs, counted once rather than twice for model arms,
there are zero crossmatch rows without shape matches;228,683 shape-catalogue
rows without crossmatches are excluded.94,712,853 parent rows lack crossmatches
and are not treated as usable detections. No usable row has a zero shape.
Detailed matching totals remain in the audit and individual output receipts.

## Decision and limits

Retain both models as experimental evidence; do not promote the extra-loss
endpoint as a general calibration fix. No production preset was changed.
These scenes belong to existing development data; this is not independent
acceptance or a robust0.3% result. Finite training length, finite common draws
and checkpoint dependence remain limitations.

The next justified experiment should supervise or constrain the joint selected
shape response alongside passing mass, using training-only targets and a
matched control. Keep full normalized response checks on both axes and all
nine cuts, rather than treating frozen shape weights or a better cut score as
sufficient preservation. This is a next hypothesis, not a demonstrated fix;
no new training was launched in this completion step.

Broad array17788286 remains held; automated follow-ups remain disabled.
[Fixed protocol](rule.md#authorized-question-and-fixed-models), [jobs](jobs.json).

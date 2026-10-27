# Symmetric F8 response investigation

## Authorization and separation of evidence — October 27, 2026

Owner requested autonomous investigation after the symmetric candidate's
[failed forward comparison](SYMMETRIC_FLOW_FORWARD.md#completed-result--october-27-2026).
Keep C8, disk emulator and production model identities fixed. No empirical
response scaling or offsets. ConstGold diagnoses the failure; it does not
select training checkpoints or tune a target to the observed1.588% bias.

Diagnostic study:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/f8_signsym_diagnosis_20261027_v1`.
Maximum two concurrent GPUs including related jobs, with one GPU per job.

## Completed stage attribution

Diagnostic17519143 completed all three stages on the same four cases:

| Checkpoint | m with C8 (%) | Flow contribution to R11 | Disk contribution to R11 |
| --- | ---: | ---: | ---: |
| Retained F8 control | +0.068015 | 0.48184046 | 0.19013467 |
| Symmetric NLL-only | +0.624179 | 0.47869139 | 0.18956964 |
| Symmetric base guard | +0.076369 | 0.48206150 | 0.18985754 |
| Symmetric final | +1.629657 | 0.47186149 | 0.18978807 |

The later harmonized refinement lowers the total response by0.01026948;
0.01020001 comes from the flow contribution and0.00006948 from disk transport
(about99.3% of the change is in the flow contribution). These include the
joint selection and own-leg normalization, not isolated derivatives at fixed
measured outputs. This locates the dominant regression in a training stage;
it does not establish that symmetry itself causes the calibration failure.

The harmonized matched-pair validation guard improves from0.319413 to0.014221
while this external response worsens. The follow-on per-leg objective uses a
different usability population; before the new backtracking experiment it
made no accepted update. Objective improvement is therefore insufficient
evidence of transfer to ConstGold. No numerical response offset is applied.

Full80-case base-guard check is queued in
`sbsi_caches/f8_signsym_guard_forward_20261027_v1`, GPU array17519273 (%1),
CPU summary17519274. The checkpoint was selected by the original training
rule before this investigation. Its external result remains a diagnostic of
the stage regression, not an independent checkpoint-selection or acceptance set.

## Frozen diagnostics

1. Compare the NLL-only, base guard and final symmetric checkpoints on cases
   40,60,80,100. Preserve64 CRN draws, original seeds/batches and selections.
   Separate flow and disk contributions using the same total selected
   denominators; require their sum to reproduce the full response. These four
   cases are for stage attribution, not calibration acceptance or selection.
2. Compare old and all three new checkpoints' likelihoods on the first four
   cases in the existing harmonized validation split:4,96,145,81, both legs.
   Convert standardized log densities to physical e1/e2/radius/flux volume
   using each checkpoint's own target-scale Jacobian before comparison.
3. On exactly those validation rows, compare the old density with its exact
   four-reflection mixture. This diagnostic establishes a symmetric reference
   without retraining; it is not an adopted multi-evaluation inference wrapper.

Implemented in `scripts/diagnose_sign_symmetric_flow.py` and
`scripts/audit_flow_reflection_projection.py`. The study's component evaluator
is a saved, instrumented copy of the completed forward evaluator; it changes
the fixed case list and records the additive response components. Original
parent/measurement/C8/pair provenance checks remain. All study inputs and
checkpoint hashes are frozen in `diagnosis_protocol.json` and per-stage
`protocol.json` files. Diagnostic GPU job17519143; CPU projection17519194.

## Training-only backtracking correction

The final joint-refinement stage originally tried six learning rates, ending
at3.125e-7. Initial training selection mass was0.2589262278666387 against
target0.2589120920527701. Even that smallest trial gave0.25888824380140113:
it overshot and increased the absolute error. Consequently no step was accepted.

Job17519144 tests12 halvings instead of6, down to4.8828125e-9, starting from
the same harmonized checkpoint with the original reference banks, gradients,
seeds, losses, full-density gates and soft/hard selection-probability gates.
No acceptance threshold is relaxed. This change follows the training
line-search receipts, not the ConstGold bias. It may still retain step0 if
no step passes all gates. Original code and failed attempts are preserved;
`refine_backtracking.py` records the exact one-line change and its source hash.
Startup: baseline NLL/guard replay exactly. All original six rates reject;
the first new rate1.5625e-7 passes training and full validation gates and is
selected as step1. Its validation guard improves0.032597→0.029495; full NLL
changes−4.702568→−4.702501, inside the unchanged0.001 budget. Step2 accepts
7.8125e-8 at the training gate; final selection remains pending.
CPU screen17519337 follows completion and symmetry audit;
an accepted selected checkpoint may trigger the same80-case external replay,
dependent on the base-guard array to respect the combined GPU ceiling.

## Validation density evidence

The four-case, both-leg projection audit completed. Weighted physical NLL is
3.30713809 for retained F8 and3.28763008 for its symmetric mixture: improvement
0.01950800 nat per row. Each new symmetric stage instead loses approximately
0.04 nat per row relative to retained F8 on these same cases. This indicates
the observed fit loss is not an unavoidable penalty for enforcing the symmetry.
It does not prove which architecture or optimizer change will recover it.
CPU/GPU evaluations differ at approximately1e-7 nat, far below this difference.

## Logarithmic invariant feature candidate

The bounded feature map psi(v)=v/hypot(1,v) compresses large shape coordinates
near±1. Added an explicit optional `invariant_features=asinh` map: replace psi
with asinh before squares and same-component products. It grows logarithmically,
retains tail distinctions and avoids raw quadratic growth. The affine shift
still uses the original signed conditioning component. Dimensions, layer count
and single-call evaluation are unchanged; both independent sign symmetries
remain exact. This is a fitting hypothesis, not an established cause or fix.

Old checkpoints default to `bounded`; saving rejects a feature-map/config
mismatch. Tests cover both maps' nonidentity density, gradients, Jacobian,
transport, optimizer updates, checkpoint round trips and tail gradients.
Full-suite job17519213 completed: **673 passed,4 skipped**, one existing NumPy
deprecation warning. Four additional tests of the automated screen passed:
failed density/guard/probability gates submit nothing, and a passing
backtracking screen retains the frozen runtime and GPU-limit dependency.
The setup script records the selected feature map in training configuration
and experiment metadata. A fresh candidate must keep the original training
data, seeds and density/response selection rules; ConstGold is not a selector.

## Autonomous candidate chain

Study `sbsi_caches/f8_signsym_asinh_20261027_v1` preserves the full original
training chain and seed611, changing only the invariant feature map. Its
source snapshot and task-local environment are separate from earlier runs.
Label-reuse17519333 passed with exact retained bytes. Base17519334 waits for
backtracking17519144 to end, maintaining the two-GPU combined ceiling;
refinement17519335 follows base/labels, then CPU screen17519336. The scheduler
had already removed completed test17519213 from its live dependency records;
its `sacct` COMPLETED/0:0 receipt was checked and saved before submitting.

The base job first replays the historical gradient-failure batch with both
bounded and asinh maps, runs architecture tests and the real-context symmetry
preflight, and stops on any failure. All model-dependent refinement references
are recomputed. `scripts/screen_sign_symmetric_candidate.py` permits the
predeclared external80-case replay only if full harmonized validation NLL
improves at least0.005 nat/row over bounded final and the per-leg guard does
not worsen. Target-coordinate scales must match and final symmetry must pass.
These are research screening rules, not production acceptance. A failed screen
is retained without an external run. Passing screens launch one GPU at a time;
no stage automatically promotes a model.

One-shot CPU receipts in `checks.json` cover two/four hours, dependent-stage
startup and completion. They save scheduler state, log tails, training progress
and any generated summaries, including subsequently submitted external arrays.
These receipts are not automatic chat notifications. The long jobs continue
under their frozen dependencies once normal startup has been verified.

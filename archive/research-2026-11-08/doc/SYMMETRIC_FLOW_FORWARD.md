# Symmetric F8 forward-model response on measured simulations

Follow-up: the failure below concerns the final refined checkpoint. The
[completed stage investigation](SYMMETRIC_FLOW_DIAGNOSIS.md#completed-follow-ups--october-27-2026)
finds that the earlier symmetric base-guard checkpoint preserves the original
80-case result: +0.027957% ±0.264014 percentage points. Neither result is
independent acceptance.

## Completed result — October 27, 2026

All80 cases and the aggregate summary completed successfully. The symmetric
flow **fails the approximately0.3% forward-response target** in this comparison.

| Flow and usability weights | m (%) | Case-bootstrap SE (percentage points) | 95% interval (%) |
| --- | ---: | ---: | --- |
| Retained F8 + C8 | +0.020770 | 0.265152 | [−0.502131,+0.533644] |
| Symmetric F8 + C8 | **+1.587790** | **0.268023** | **[+1.058476,+2.107470]** |
| Retained F8 + true usable flags | +0.228311 | 0.266074 | [−0.295695,+0.745195] |
| Symmetric F8 + true usable flags | +1.810052 | 0.268763 | [+1.279010,+2.331723] |

The paired new-minus-old change with C8 is **+1.567020 ±0.010556 percentage
points** (case-bootstrap SE). The measured response is0.6707665225; the new
model predicts0.6602826210, compared with0.6706272357 for retained F8.
The discrepancy persists when the simulation's true usable flags replace C8,
so C8 alone does not explain it. This localizes a problem to the new flow with
the fixed disk emulator and measured selection; it does not isolate which
training stage or modeling constraint caused it.

The largest change in magnitude-bin contributions is in true magnitude24–25.5:
−0.855839 to+0.534613 percentage points. These are contributions using the
total selected normalization, not biases independently normalized within bins.
Selected counts remain close: new C8 model7,262,565/7,262,804 versus measured
7,264,215/7,263,821 in plus/minus legs. Matching counts and exact flow symmetry
therefore do not establish accurate shear response.

Validation: all eight GPU tasks and CPU summary17518244 exited0; combined GPU
allocation time87m24s on one H200 at a time. All80 case receipts match the
frozen protocol. The retained bias and SE reproduce exactly. Results are in
the study's `summary.json` and `cases/case*.json`; original logs are retained.
Cancelled the redundant pending two-hour check17518301 after final review.
No model promotion, new training, or posterior inference was launched during
this review. Next investigation should distinguish density fitting from
response refinement while keeping this failed comparison as test evidence.

## Request and frozen comparison — October 27, 2026

Owner requested checking whether the newly trained model still delivers a
forward response within approximately0.3%. This study compares symmetric F8
with the retained F8, keeping primary C8 and the physical-moment disk emulator
unchanged. It is forward-model response validation, not posterior sampling.
The comparison uses the original antithetic endpoints, not a one-sided0→g
derivative. See [response definitions](CONVENTIONS.md#6-response-estimators).

Study: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/f8_signsym_forward_20261027_v1`.
Training candidate: [completion results](SYMMETRIC_FLOW_TRAINING.md#completion-results--october-27-2026).
Candidate SHA256: `05dc7a986610851208f10d3a336a7805c5b2a643e3e16f9ed3ca1f8a0d33fd57`.
No fitting, checkpoint selection, or production model changes occur here.

## Population, response and uncertainty

- ConstGold cases40–119, generated target-role parent, g1=±0.02.
- Independently select each measured leg: usable ngmix shape, measured radius
  >0.6arcsec and magnitude<25.8. No truth-property or extra ellipticity cut.
- Model: unchanged C8 usable probability times joint F8 selected moments;
  disk transport uses the radius and flux of that same draw. Each leg has
  its own selected normalization. A second diagnostic substitutes the
  simulation's own usable flag for C8 on each leg.
- 64 common-random-number draws, batch1024, original latent seed7301 plus the
  original case/block offsets. The new flow and disk predictions are regenerated.
- R=(mean_plus−mean_minus)/0.04 on both sides;
  m=100*(R_sim/R_model−1)%, per
  [bias conventions](CONVENTIONS.md#7-multiplicative-bias).
- 10,000 paired case-bootstrap resamples, seed20260925; report standard error
  in percentage points,95% interval, and paired change from retained F8.
- Four magnitude bins split at24,25.5,27.5 provide response-error contributions
  using total selected denominators. Contributions sum to total m; their
  absolute sum diagnoses cancellation. They are diagnostics, not truth cuts.

## Provenance and checks

`scripts/evaluate_sign_symmetric_forward.py` deliberately uses the immutable
training snapshot environment and historical evaluator helpers. It does not
import an archive into the active library. `protocol.json` pins99 input/source
files; each job checks their hashes and the training snapshot's frozen-input
receipt. Parent and saved original per-row data are checked against their
original manifest hashes. Identity joins must match exactly and parent IDs
must be sorted and unique. Pair loading retains its original provenance checks.

Measured outputs and original model moments are reused from the original
hash-verified evaluation rows. Their original `evaluation/chunk*/results.json`
receipts retain per-case unmatched-row/usable counts; unmatched measurements
remain excluded. This is not a new raw-catalogue measurement or full-data audit.
Cached C8 probabilities are hash-pinned and revalidated against the original
C8 evaluation for every case, leg, all three original cuts and all four bins.
The aggregate summary must reproduce the published retained control
**+0.020769635769424255% ±0.26515217441748873 percentage points** within1e-9,
or fail without publishing a summary. No empirical adjustments are applied.

Small summary tests verify unequal per-leg denominators, bias sign, paired
differences, additive bin contributions and rejection of a wrong control:
`python -m pytest tests/test_sign_symmetric_forward_summary.py -q`:
**2 passed**. Python syntax, shell syntax and `git diff --check` passed.

## Launch and monitoring

This section preserves startup history; the [completed result](#completed-result--october-27-2026)
supersedes its pending status.

`jobs/job_sign_symmetric_forward.sh` runs one H200 at a time,8 allocated logical
CPUs,48GiB host memory. No other user jobs were active at submission.
First chunk17518242 covers cases40–49. Array17518243 covers the remaining
seven chunks with throttle1 and depends on successful completion of the first.
CPU summary17518244 depends on both arrays. Invalid dependencies cancel.

Startup verified frozen inputs and loaded the model on an H200 NVL. Case40
completed349,784 parents in71.5s; its C8 replay and identity checks passed
before draw generation, and case41 is progressing. At this initial pace the
full run should take approximately1.5–2hours plus queue delays. The project
filesystem has14TiB available. No aggregate calibration result is available yet.
`checks.json` schedules one-shot CPU status receipts at two hours, first-chunk
termination and final-summary termination. These save scheduler/log snapshots;
they are not automatic chat notifications or scientific acceptance decisions.

## Interpretation limits

These are reused development scenes and g1 only. Case-bootstrap errors exclude
training and finite-draw uncertainty. A point estimate within0.3% does not
establish robust absolute calibration; the retained control itself has a
roughly0.265-percentage-point standard error. This response test also does not
establish removal of c2 in posterior inference. C8 remains unchanged.

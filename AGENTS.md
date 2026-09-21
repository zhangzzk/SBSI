# Agent instructions for SBSI

SBSI is a general shear-calibration library with one inference workflow.
`v1.1-infer` names the default numerical pipeline, `v1.2-infer` names the
tilted-stratified trimmed sampler that supersedes its estimator but is not yet
the default, and `v3.2-like` names the likelihood composition both share.
`V3.6-like` names the retained single common-descent flow, physical-moment
disk-response emulator with Möbius composition, and single 17-input
smooth-crowding usable-event classifier. Its artifact identity is pinned in
`configs/models_v3_6_like.json`. The owner closed the investigation on
2026-09-20 and cancelled the unfinished independent validation. Existing
development results are retained; no independent result was produced.
`V3.5-like` remains the distinct historical epoch154/trial9/three-classifier
model preset. Neither model preset changes the configured likelihood.

All project documentation lives in `doc/`. Only this file, `CLAUDE.md`, and
`README.md` stay at the repository root:

- `doc/API.md` — the BlendEMU scope boundary, the general API contract, and the
  public API.
- `doc/INFERENCE.md` — the current inference/likelihood release boundary and
  the estimator derivation.
- `doc/CATALOGUE_PRIOR.md` — the finite scene prior, likelihood, caches, and
  scalable importance-sampling workflow.
- `doc/MATH.md` — the detailed Bartlett/score derivation used by the inference
  implementation.
- `doc/CONVENTIONS.md` — the definitions every reported number must cite:
  catalogues, population-cut order, shape kinds, response estimators, seeds,
  and the definition of `m`.
- `doc/ENVIRONMENT.md` — interpreters, running the test suite, the BlendEMU
  import boundary, and the login-node/scheduler policy.
- `doc/WORKLOG.md` — the dated record of every substantive change, newest
  first, linking to preserved historical logs.
- `doc/JOINT_CALIBRATION.md` — V3.6-like identity, evidence, and stopping point.
- `doc/REPOSITORY.md` — active layout, archive inventory, and restoration rules.
  Completed experiments are preserved under `archive/research-2026-09-14/`
  and `archive/research-2026-09-20/`; they are not active documentation.

`doc/` is pulled on demand, and by section, not cover to cover. Read headlines first and then the relevant entries. When writing a doc, keep it navigable this way: greppable headings, and cite sections by anchor, not by file alone.

## Rules

- Classifiers, conditional flows, response prediction, validation, and Bayesian
  inference stay in SBSI; BlendEMU owns image simulation, measurement,
  simulation-catalogue production, and emulator training. Boundary detail:
  `doc/API.md`.
- Active code covers model loading/prediction, likelihood and inference. Do not import from
  `archive/`; restore an archived experiment deliberately if new work requires
  it.
- Unmatched rows between catalogues should be dropped or rejected and be reported.
- No empirical offsets, scaling factors, or pasted results to make a number
  agree with an expectation. Use common random numbers when differencing
  stochastic flow evaluations, and apply the same forward or antithetic
  extraction convention to simulation and model sides (`doc/CONVENTIONS.md`
  §6c–6d).
- The login node is for edits, syntax checks, and other light work; other nontrivial heavy CPU/GPU work go through the local
  scheduler. Interpreters and test commands: `doc/ENVIRONMENT.md`.
- The ConstGold/joint investigation was stopped on 2026-09-20. Do not restart
  its archived launchers or recovery chain without a new owner request.
  For future authorized work, the owner's 2026-09-14 limit remains
  **two GPUs total**, including the 500k inference, 20k numerical study, and
  related diagnostics. Check other active/pending SBSI GPU jobs and use
  dependencies and array throttles to preserve this combined limit.
- Monitor newly submitted jobs through startup and their first meaningful
  progress so early failures are caught and repaired before yielding. Brief
  status/log checks roughly every 30--60 seconds are allowed during startup;
  short preparation, validation, or test jobs (normally under 30 minutes) may
  be followed through completion. Check each dependent stage when it starts.
  Once a long job is progressing normally, or queue/startup waiting becomes
  prolonged, yield and use a scheduled one-shot check about every two hours;
  do not occupy a task polling a long run until it finishes. Report meaningful
  progress, failures, or required action; avoid repetitive unchanged-status
  messages. Owner-requested immediate checks are always allowed. This applies
  to goal-mode continuations too and supersedes older blanket no-polling notes.

## Work log

After every substantive change, prepend a dated entry to `doc/WORKLOG.md` covering
files and behavior changed, validation commands and outputs, limitations, and
next steps.

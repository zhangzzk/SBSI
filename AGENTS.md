# Agent instructions for SBSI

The current GalSBI fiducial is `like-1.0-galsbi`, selected by the owner on
2026-11-08. Pin `neighbour_conditioned_flow_20261106/control/nine.pt`
(SHA256 `9bc10a6548e792cdcbb29b226e20e4e53dff74d88c1a29a972e246cb07227417`),
the GalSBI C8 classifier and disk emulator in `configs/models_like_1_0_galsbi.json`.
This is the factorized NLL-plus-weak-nine-cut control with neighbour score
disabled, not `neighbour_nine`, original `nine_weak`, or latest weighted flow.
Four-scene paired-seed development results: m1=+0.38554 ±0.39755%,
m2=+0.04296 ±0.66616%, nine-cut RMS=0.53431%. No independent acceptance or
full40 paired-seed evaluation of this checkpoint. See `doc/LIKE_1_0_GALSBI.md`.
Naming this fiducial does not change the configured likelihood or resume jobs.
The retained multistage comparison is `galsbi-multistage` (flow43673e5d), pinned
in `configs/models_galsbi_multistage.json`. Both models and shared companion
artifacts are included under `models/`; load through `load_measurement_model`.
Current continuous inference is `scripts/run_galsbi_inference.py`, integrated
from the October production cache-off sampler. It requires matching model
identity in reference metadata. The multistage model requires an extra
reference-neighbour summary and is explicitly rejected by this inference
entry until that adapter is validated. See `doc/CURRENT_INFERENCE.md`.


SBSI is a general shear-calibration library with one inference workflow.
`v1.1-infer` names the default numerical pipeline, `v1.2-infer` names the
tilted-stratified trimmed sampler that supersedes its estimator but is not yet
the default, and `v3.2-like` names the likelihood composition both share.
`V3.9-like` is the retained F8 flow, C8 usable-event classifier (first
predeclared seed20260913), and original physical-moment disk emulator with
Möbius composition. Its identity is pinned in `configs/models_v3_9_like.json`.
The owner wrapped up this investigation on 2026-09-25. Main development bias
is +0.020770% ±0.265152 percentage points after independent per-leg measured
radius>0.6 arcsec/magnitude<25.8 cuts, without truth-property cuts. Broader
classifier checks failed; no independent acceptance or robust 0.3% claim.
No emulator simplification was run. See `doc/V39_MODEL.md`.
The dedicated disk inference preparation remains V3.6-specific; do not
relabel old 17-input feature stores as V3.9.

`V3.6-like` names the historical single common-descent flow, physical-moment
disk-response emulator with Möbius composition, and single 17-input
smooth-crowding usable-event classifier. Its artifact identity is pinned in
`configs/models_v3_6_like.json`. The owner closed the investigation on
2026-09-20 and cancelled the unfinished independent validation. Existing
development results are retained; no independent result was produced.
`V3.5-like` remains the distinct historical epoch154/trial9/three-classifier
model preset. No model preset changes the configured likelihood.

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
- `doc/V39_MODEL.md` — retained September model identity, features, evidence and stopping point.
- `doc/LIKE_1_0_GALSBI.md` — current GalSBI fiducial identity, evidence and loading boundary.
- `doc/JOINT_CALIBRATION.md` — historical V3.6-like evidence and stopping point.
- `doc/REPOSITORY.md` — active layout, archive inventory, and restoration rules.
  Completed experiments are preserved under `archive/research-2026-09-14/`,
  `archive/research-2026-09-20/`, and `archive/research-2026-09-25/`; they are not active documentation.

`doc/` is pulled on demand, and by section, not cover to cover. Read headlines first and then the relevant entries. When writing a doc, keep it navigable this way: greppable headings, and cite sections by anchor, not by file alone.

## Rules

- The F8/C8 investigation is closed by owner decision 2026-09-25. Do not
  restart its archived launchers or begin emulator simplification without a
  new owner request. Preserve V3.9 artifact identities and reported limitations.

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
- The owner's 2026-11-06 scheduling preference is **fewer, larger, longer-lived
  jobs**, rather than many jobs for small chunks. Pack multiple cases/legs into
  each allocation using persistent workers or a bounded worker pool; reuse
  loaded models when practical. Preserve per-case seeds, outputs and restart
  checkpoints, pilot gates, dependencies and the authorized CPU/GPU ceilings.
  Job packing does not imply increasing numerical batch sizes. See
  `doc/ENVIRONMENT.md` §Job packing.
- The ConstGold/joint investigation was stopped on 2026-09-20. Do not restart
  its archived launchers or recovery chain without a new owner request.
  For future authorized work, the owner's 2026-09-14 limit remains
  **two GPUs total**, including the 500k inference, 20k numerical study, and
  related diagnostics. Check other active/pending SBSI GPU jobs and use
  dependencies and array throttles to preserve this combined limit.
- On 2026-11-05 the owner authorized more GPUs for the current factorized
  GalSBI constant-shear evaluation. This study now uses up to **six GPUs total**
  across `cip` and `inter`, superseding its earlier two-GPU limit. Keep the
  existing evaluation array throttled to six, respect partition limits, and
  preserve cases, seeds, model identities and report dependencies.
- On 2026-11-06 the owner authorized up to **500 CPUs** for the current
  constant-shear simulation study. Pack independent case/leg workers to avoid
  the scheduler's per-user job limit; preserve pilot validation gates and all
  scientific settings. Allocate only as many workers as the remaining work
  can use. This does not raise the study's six-GPU limit.
- On 2026-11-08 the owner reaffirmed **up to 500 CPUs** for the current
  paired-seed measurement and calibration continuation. Increase useful CPU
  concurrency within that combined ceiling, preserving completed checkpoints,
  seeds, scientific settings and report dependencies. Prefer a few packed
  allocations on fast-I/O nodes. This does not resume the held broad evaluation
  array or change GPU limits.
- For the current Cartesian GalSBI flow study, the owner clarified on
  2026-11-06 that internal cancellations are acceptable. Prioritize response
  robustness near the target: radius>2.8/3.0/3.2 pixels crossed with
  magnitude<24.9/25.0/25.1 (nine joint cuts). Distant-cut and subpopulation
  errors are diagnostics, not automatic rejection criteria for this target.
- On 2026-11-06 the owner paused the remaining broad one/nine-cut constant-shear
  evaluations while focused alignment and neighbour-response localization
  continues. Keep pending array17788286 held until an owner request to resume;
  preserve completed outputs and allow already running cases to finish.
  This pause does not stop the focused diagnostics or raise the six-GPU cap.
- On 2026-09-25 the owner explicitly authorized more GPUs for the V3.9-like
  model-generated sampling study (20k and the authorized100k/500k follow-ups).
  Its inference array permits six
  concurrent GPUs across `cip` and `inter`, superseding the two-GPU limit
  for this study.
  Preserve stage dependencies and check other study jobs to stay within
  this six-GPU ceiling. Scientific settings and random seeds stay fixed.
- On 2026-09-26 the owner requested **four persistent jobs, one GPU each**
  for the current500k continuation. Keep this continuation and its related
  diagnostics within four concurrent GPUs. Preserve original four-observation
  likelihood groups: larger density batches failed the numerical replay check.
  Atom retrieval and proposal draws may use separately validated larger batches.
- On 2026-10-01 the owner authorized the measurement-initialization comparison
  on two `inter` GPUs alongside the two old continuous-prior MH runs on `cip`.
  This study and its performance diagnostics may use **four GPUs total** across
  both partitions, superseding the earlier two-GPU cap for this work. Keep the
  initialization array throttled to two; retain dependencies needed for actual
  prerequisites or the combined resource limit, not unrelated scientific work.
- On 2026-09-24 the owner explicitly authorized more GPUs for the F8
  simplification experiment, superseding its historical two-GPU limit. Its
  evaluation array permits six concurrent workers; the `cip` partition QoS
  limits allocations in that partition to three per user. The owner also
  explicitly authorized `inter`; F8 may use both partitions within its six-worker
  ceiling and the scheduler limits.
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

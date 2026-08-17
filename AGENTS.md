# Agent Instructions For SBSI

This directory is a standalone SBI/shear-calibration project.

## Scope

- Keep SBSI implementation code under `SBSI/`.
- Treat `blendemu` as the place where simulation outputs and catalogue-building live.
- In SBS, consume completed blendemu catalogues; do not add catalogue builders here unless the user explicitly changes this boundary.
- Do not move SBS-specific classifier, flow, validation, or inference code into `blendemu` unless the user explicitly asks for that integration.
- If using blendemu outputs, read them as input data and write SBSI-derived products under `SBSI/data`, `SBSI/models`, or another user-specified SBSI path.

### The blendemu boundary (enforced)

The user runs blendemu themselves and hands SBSI a catalogue path. Two crossings exist,
and they are the only ones allowed:

1. `sbs_shear/emulator.py` — the *sole* module that imports blendemu. It is needed only
   to evaluate `R_blend` via `BlendingPredictor`, and it imports lazily so the rest of the
   pipeline runs with blendemu absent. Scripts must call `load_blending_predictor()`
   rather than importing blendemu, editing `sys.path`, or hardcoding a models directory.
2. `scripts/build_detection_measurement_catalogue.py` — the optional catalogue bridge.

Rules for new code:

- Never write an absolute path into a script. Add a root to `sbs_shear/paths.py` and use
  it; every root is environment-overridable (`SBSI_CATALOGUE_DIR`, `SBSI_CACHE_DIR`,
  `SBSI_SIM_DIR`, `SBSI_CONST_SIM_DIR`, `BLENDEMU_ROOT`, ...).
- Never re-declare the survey conditions. Import `SURVEY_CONDITIONS` / `RESCALE_KW` from
  `sbs_shear.emulator`.
- Jobs that `cd` into blendemu belong in `jobs/blendemu_side/`, not `jobs/`.

## Work Log Requirement

- After any substantive SBSI change, update `SBSI/WORKLOG.md` before the final response.
- Add a dated entry with:
  - files added or changed,
  - what behavior or workflow changed,
  - commands run for validation,
  - important outputs or known limitations,
  - next recommended steps.
- Keep entries concise and factual. Do not paste long terminal logs.

## Resource Policy

- Use Slurm jobs for work that meaningfully consumes CPU, GPU, memory, or wall time:
  training runs, full-catalogue scans, multi-case simulations, response production,
  and large validation jobs.
- Local commands are acceptable for negligible work only: file edits, syntax checks,
  notebook JSON validation, metadata inspection, and tiny smoke tests.
- Current full-catalogue selection jobs are expected to request nontrivial
  resources. Use the existing job scripts as the baseline: training and feature
  importance use 250G/16 CPUs/1 GPU; blend, gradient, and invariance diagnostics
  use 128G/12 CPUs/1 GPU.

## Development Notes

- Prefer standalone model, validation, and inference utilities in `SBSI/sbs_shear`.
- Prefer runnable scripts in `SBSI/scripts`.
- Keep generated data and model artifacts out of source files; use `SBSI/data` and `SBSI/models`.
- Keep all orientation-dependent features in the SBS canonical spin-2 basis
  `(q1, q2) = q(cos 2 theta, sin 2 theta)`. Blendemu catalogues use this same
  basis; missing `shear_component_convention` metadata should not trigger a
  shear-component relabeling.
- The first model target is the differentiable selection classifier:
  `P(s=1 | true properties, neighbour properties, shear)`.
  The current pilot uses SExtractor detection as `s=1`.
- For nearest-neighbour pair-frame selection models, use `neighbored` as the
  close-blend/rendered-neighbour indicator. Do not add a redundant
  `within_blend_radius` model input unless a later analysis proves it is needed.
- When every row has a nearest-neighbour frame, validate that `neighbored=False`
  rows are not spuriously sensitive to far-neighbour secondary or pair-angle
  properties before trusting the model response.
- The current default selection model is the primary-major-axis frame model
  trained on the capped `sbs_skycos` catalogue. It avoids assigning an arbitrary
  nearest-neighbour direction to isolated rows while keeping secondary and
  pair-geometry features gated by `neighbored`.
- For primary-frame models, check gradient and performance summaries as a
  function of `e_abs_p`; the frame becomes physically weak for nearly round
  primaries even though the numerical fallback is deterministic.
- Gradient validation against finite differences is required before treating `dP/dgamma` as a science result.
- The next model target is the selected-object measurement likelihood
  `p_meas(xhat | true properties, neighbour properties, shear, s=1)`.
  Keep it factored separately from the selection classifier; compose the two
  only when constructing the unnormalized catalogue density.
- For scene-level measurement models, group pair-annotated detection catalogues
  by `(case, shear_case, input_index)` and condition on the set of all annotated
  neighbours inside the aperture. This only represents the intended
  all-neighbour scene if the source blendemu catalogue was built with `k` large
  enough to cover that aperture.
- Keep full-geometry and radial-only scene models as explicit ablations.
  `--geometry-mode full` keeps pair-angle/oriented features; `--geometry-mode
  radial` keeps neighbour separation and scalar properties only.

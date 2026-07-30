# Agent Instructions For SBSI

This directory is a standalone SBI/shear-calibration project.

## Scope

- Keep SBSI implementation code under `SBSI/`.
- Treat `blendemu` as the place where simulation outputs and catalogue-building live.
- In SBS, consume completed blendemu catalogues; do not add catalogue builders here unless the user explicitly changes this boundary.
- Do not move SBS-specific classifier, flow, validation, or inference code into `blendemu` unless the user explicitly asks for that integration.
- If using blendemu outputs, read them as input data and write SBSI-derived products under `SBSI/data`, `SBSI/models`, or another user-specified SBSI path.

## Fiducial Model (set 2026-07-30)

**The fiducial model is the V2 dom6x6 flow + the tuned in-domain blend emulator.** The prediction is
`R_model = R_flow + R_blend` — the flow alone is SELF-response only and is never the whole model.

- **Flow:** `measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s{seed}_swaavg.pt` under
  `/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/`. 16 checkpoints: 501, 502, 503,
  505–517 (**504 does not exist**). Trained domain: primary true `mag < 26`, `Re > 0.3`.
- **Emulator:** blendemu tag `lsst_r_extnbr_indom_tuned` (73-trial Optuna search). Per-object lookup:
  `SBSI/results/blend_lookup_indomtuned_c40-139.feather` (`case`, `input_index`, `R_blend`).
- **constgold, in-domain, no cut, 16 seeds:** `m = -0.123 +- 0.152%` (std 0.608%),
  `R_sim = 0.8605`, `R_blend = 0.1358`, N = 11,674,408. This is the shape-standard number; a 4-seed
  subset gives ~-0.5% because s503 is a -1.4% outlier, which is exactly why shape numbers use 16.

### Two traps — both fail SILENTLY, neither raises

1. **The tuned emulator is IN-DOMAIN ONLY.** It applies its stored training cuts (mag 18–26,
   Re 0.3–1.5) at inference and returns nothing outside them, so wide-population rows fall back to
   `R_blend = 0`. On the wide population it covers only **43.4%** of rows, collapsing `<R_blend>` from
   0.1593 (certified `lsst_r_extnbr_ho`) to 0.0589 and producing a spurious **+28.9%** m. It cannot
   replace `_ho` for wide-population work. **Always assert the lookup match fraction before using it**
   and drop unmatched rows rather than zero-filling them.
2. **Gold-v1 and dom6x6 are not interchangeable by population.** Gold-v1
   (`meas_szfl_noz_lam450_fixresp`) was certified WIDE (+0.245%); dom6x6 is the in-domain flow. On the
   cut population with identical `R_blend = 0.1358`: Gold-v1 `R_flow = 0.6836 -> m = +4.74%`;
   dom6x6 `R_flow = 0.7268 -> m = -0.50%`. The cut population needs `R_flow = R_sim - R_blend =
   0.7224`. **A ~+5% in-domain m is the signature of using Gold-v1 out of its domain**, not an
   emulator fault.

Optuna tuning of the emulator is a **null** on constgold m: +0.010 pts, sd 0.000 across seeds, ~20x
below seed error. Emulator promotion is argued on the per-pair ruler (`scripts/eval_rblend_gap.py`),
**never** on constgold m — constgold is evaluation-only (the R_blend firewall).

## Numerical Integrity

- **No silent empirical or hardcoded corrections.** Do not apply a numerical offset, scale factor,
  fudge, or "calibration" constant to any reported quantity unless it has been explicitly discussed
  and agreed with the owner.
- This covers: offsets carried over from an older analysis, constants pasted from a previous run,
  multipliers that make a number land where it is expected, and any figure or table that mixes a
  freshly computed quantity with a pasted one without saying so.
- **If a quantity cannot be computed properly, do not substitute a number.** State that it is
  blocked, state what would unblock it, and leave it out. A caveat in a docstring or figure caption
  is NOT licence to ship the fudge — deleting is preferred to captioning.
- If a correction genuinely is needed, raise it first; once agreed, make it explicit in the output
  (named, printed, provenance-stamped), never folded silently into a plotted or tabulated value.
- This does NOT forbid parameters derived and reported within the same run (e.g. proxy-fit
  coefficients quoted alongside their residual error). The line is: derived-and-reported here = fine;
  pasted-in-from-elsewhere and unlabelled = not.
- Trigger case (2026-07-31): V1's `fig4_bias_prob_neighbours` applied `R_blend + 0.0017`, an offset
  inherited from a 2026-07-09 forward-model residual, because a faithful run was blocked. It was
  DELETED rather than regenerated under the new fiducial model.

## Ensemble Seed Convention

- **Shape-related results use 16 flow seeds; selection-related results use 4.**
  Shape-related = global/no-cut `m`, certified-pipeline numbers, any absolute multiplicative bias.
  Selection-related = `m_sel` / `m_flow` under measured cuts.
- A 4-seed selection table is the intended standard, not an under-powered one. Do not quote a
  shape/global `m` from 4 seeds.
- Rationale: selection quantities are model-vs-sim ratios at the SAME cut from the SAME draws, so the
  common-mode seed offset largely cancels; an absolute shape `m` has no such cancellation.
- A table reporting BOTH (e.g. the constgold near-domain table, whose no-cut row is a global `m` while
  its cut rows are selection) must report each at its own standard and say which is which.

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

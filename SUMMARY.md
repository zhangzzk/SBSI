# SBSI Shear-Calibration — Summary

**Goal:** multiplicative shear bias `m` sub-percent (Stage-IV |m| < 0.3%), *legitimately* —
forward-model fidelity, **no empirical m-removal**. **Setup:** train `p(measured | true, neighbour)`
+ selection `P(s=1)` on a **g=0** catalogue; shear enters only via the analytic Möbius map `S_γ`
on the true shape; recover by marginal-likelihood scan on held-out sheared catalogues (g=0.02/0.05/0.2).
Heavy work via Slurm. Full history in `WORKLOG.md` (newest-first).

**Shape estimator (since 2026-07-01): ngmix `NGMIX_G1/G2`** (PSF-corrected reduced-shear ε).
Earlier work used the SExtractor windowed-moment ellipticity `(a−b)/(a+b)`, which is **PSF-diluted
and not a shear estimator** — pre-pivot numbers below refer to that estimator and are kept for the record.

## Major milestones

1. **Baseline & closure.** Flow-MLE recovery `m = +3%`; closure test unbiased (<0.2%) → machinery sound.
2. **Bias = shear-response miscalibration, not g=0 fidelity.** Seven converging diagnostics (closure,
   richer flows, p_cat, decorrelation, selection, GalSim controls, g=0 null) → `m` is a *response
   transfer gap*, decoupled from g=0 likelihood quality.
3. **Direct response calibration → sub-percent (SExtractor era).** First-moment (BFD-like) estimator with
   responsivity from the g=0.05 sim, tested on an independent g=0.02 render → `m = −0.24% ± 0.63%` (28M rows).
4. **Response-aware flow training.** Sobolev loss `L = NLL + λ‖R_model − R_sim‖²` + MLP mean head +
   property-resolved (flux×size×blend) supervision → the flow's induced response tracks `R_sim` across a
   ×4 range and across blend bins (fixed the earlier flat/inverted response). Flow-MLE slope remained ~+3%.
5. **Gold constant-shear validation** (`validate_constant_response.py`, two-sided ±0.02) → revealed the
   flow target and the gold measured **different estimators** → triggered the ngmix pivot.
6. **Estimator pivot to ngmix** (2026-07-01): re-aggregated `det_meas` with `NGMIX_G1/G2`.
7. **The 0.30-vs-0.39 isolated-response gap SOLVED = coherent blending** (not a bug).

## Key conclusions (current)

- **Calibrate ngmix, not SExtractor** — ngmix is the PSF-corrected shear estimator (isolated R≈0.39 raw
  vs SExtractor R≈0.24).
- **The constant-vs-half response gap is coherent blending.** Coherent-field shear boosts the ngmix
  response via aligned neighbour light (a low-S/N effect); incoherent per-galaxy shear (the half render)
  does not. Zero-neighbour galaxies agree (const=half); the gap grows with local density. It is **not**
  detection / crossmatch / index-join / shear-convention / ngmix-convergence / anisotropy / shear-magnitude.
- **Split of labor:** the SBSI flow models the incoherent **self-response**; the external **BlendEMU
  blending-response emulator** supplies the coherent part; combined they reproduce constant-shear.
- **ngmix "convergence" is a non-issue** — the ~50% "failures" were the `--targets secondaries`
  bookkeeping split (only the 2nd half of the input list is fit); ~100% converge where actually fit.
- **Selection = `detected`** — ngmix-convergence dropped as a selection channel (it modeled the split).

## Current plan

1. **Rebuild the flow catalogue as all-pairs** (approach A, **7″/k=20**), secondary target, ngmix —
   mirroring the BlendEMU blending-response structure with **primary↔secondary swapped**; **each
   (target, one-neighbour) pair = one data point**. (Current flow is thin: nearest pair, 3″/k=2.)
2. **Fix multi-row-per-galaxy weighting** in `compute_response_target_blend.py` + the flow trainer
   (all-pairs over-weights dense galaxies ~17:1).
3. **Retrain the measurement flow** (response-aware, self-response target) on the all-pairs catalogue.
4. **Classifier:** selection = detected only; keep nearest-pair / individual-galaxy.
5. **Validate:** self-response flow ⊕ BlendEMU blending emulator → constant gold (`m`, `c`).
- Open: aperture 7″ (×8) vs 10″ (×16) — lean 7″; whether the classifier also needs all-pairs — lean no.
- Flag (open): a secondary target's secondary neighbour is also sheared (random direction) in the
  response/validation renders → assumed to average to noise; testable as a null.

## Repo layout

- `sbs_shear/` — core library (measurement/selection/scene models, preprocessing, shear map, sim stream).
- `scripts/` — **core pipeline** only: `build_detection_measurement_catalogue`, `train_measurement_model`,
  `train_selection_response`, `compute_response_target_blend`, `compute_selection_target_blend`,
  `validate_constant_response`, `response_ratio_diagnostic` (load-bearing dep of the validator).
- `jobs/` — active Slurm jobs (build / targets / train / validate).
- `archive/`, `jobs/archive/` — one-off diagnostics & superseded jobs (reference only; kept one level under
  the repo root so their `SBSI_ROOT` path resolution still works).
- Docs: `WORKLOG.md` (newest-first), `SBI_shear.md` (method/research plan),
  `SBI_shear_response.md` (response-aware addendum), `AGENTS.md` (project scope/instructions).

# archive/ — one-off diagnostic & probe scripts

Exploratory / diagnostic scripts that produced results now recorded in `WORKLOG.md` and
`SUMMARY.md`. Kept for reference and reproducibility; **not part of the active pipeline**
(that lives in `../scripts/`).

Located one level under the repo root on purpose: these scripts compute
`SBSI_ROOT = dirname(dirname(__file__))`, so they still resolve `sbs_shear` and
`scripts.*` imports from here (verified).

Highlights from the 2026-07-01 coherent-blend investigation:
- `response_density_probe.py` — the decisive test: isolated ngmix response vs local neighbour
  count → const=half at 0 neighbours, gap grows with density (coherent blending).
- `response_single_snr.py` — const@0.02 vs half@0.02 vs half@0.05 by measured S/N (high-S/N convergence).
- `response_noise_probe.py` — E/B-mode null + noise-sharing.
- `response_angle_probe.py` — shear-axis anisotropy (refuted).
- `neighbor_shear_status.py` — a secondary target's neighbours are 50% unsheared / 50% random-sheared.
- `ngmix_convergence_probe.py` — ngmix converges ~100% where fit (the "50% fail" is a bookkeeping split).

Earlier diagnostics (response resolution, blend ablation, selection cuts, SNR sweeps, GalSim
controls, etc.) from the 2026-05/06 sessions are also here.

## 2026-08-03/04 per-object response pin + SNC label investigation (CLOSED, no improvement)

Goal was to fix flow #1's small/faint self-response defect. Both halves of that defect turned out
to be artefacts of how the number was measured, and the resulting model change did NOT improve
flow #1 — it traded a half-shear gain for a larger constgold loss. Archived on the owner's call.

- `diag_snc_estimator_gap.py`, `diag_snc_source_decomp.py` — the standing finding: the project has
  TWO SNC estimators whose g=0 partners are independent measurements of the same objects; they
  disagree by -3.06% at small size, and the disagreement grows toward small/faint galaxies. This
  result is NOT retracted; it is why any label and its score must use the same estimator.
- `diag_perobj_conditional.py` — supervision ceiling on true properties; also showed the FAINT
  defect is a MEASURED-S/N selection artefact (under a TRUE-mag cut the flow was -0.53 +- 0.57%).
- `build_perobj_target.py` / `build_perobj_oracle.py` — the per-object E[R | true props] target
  (`--label-source ruler|dump`) that replaced the per-cell grid pin.
- `diag_perobj_train_residual.py` — ruled out a population/generalisation explanation.
- `score_pin_pilot.py`, `plot_selfresp_labelfix.py`, `plot_fig2_labelfix.py` — scoring and the two
  comparison figures.

**Outcome, in one line:** label-consistent retraining closed half-shear small size (-3.74% ->
-0.08%) but moved constgold the wrong way (+2.43% -> +5.00% at small size, +0.27% -> +1.36%
overall), so it was not adopted. Full record in `WORKLOG.md` 2026-08-03p..x and 2026-08-04a/b.

**Two live leads survive this closure** (see WORKLOG 2026-08-04b): the emulator may be
over-predicting `R_blend` at small size with the flow's deficit masking it — testable on the
per-pair ruler, `scripts/eval_rblend_gap.py` — and the `Re ~ 0.31` domain-edge bin is wrong in both
evaluation sets and both models (+16% on constgold, +7% on half-shear).

NOTE: the two `plot_*_labelfix.py` files import `plot_fid_flow_figures` from `../plotting/`; run
from there or fix the path if resurrected.
## The §5C probe fleet (`diag5c_*.py`, archived 2026-08-01)

27 probes and their 31 job scripts (`../jobs/archive/job_diag5c_*.sh`, `job_lag5c_sweep.sh`) from
the investigation of `INFERENCE.md` §5C, the Lagrangian score. **Closed, not paused.** §5C's
machinery is exact against the closed forms of `INFERENCE.md` A.7 (cont.164), but on the real V2
model its integrand $\partial_\gamma\log p_{\rm flow}$ has a Hill tail index near 1.3 — below 2, so
no finite second moment — and the estimator's denominator *grows* with bank size (fitted exponent
+0.20 against −1 for honest Monte Carlo) while a tame integrand on the identical weights averages
down. cont.170 showed that survives removing both known setup defects (bank duplication, missing
`true_cut`), so the weights were never at fault; the integrand is. Read `diag5c_bankctl.py`'s
docstring first — it states the protocol and the outcome map that settled it.

`closure_v2_lagrangian.py` sits here too, alongside the probes that import it. It is the §5C
driver and also holds shared V2 plumbing (`rebuild`, `load_rows`, `scene_context`, `phi_block`).
An earlier version of this note said it was deliberately kept in the active `scripts/` tree so
V2-side §5B work could reach that plumbing; that is no longer accurate. The live §5B drivers
(`scripts/eval_score_select.py`, `scripts/eval_score_response.py`) do not import it — checked at
the 2026-08-17 merge — and the probes that do are now co-located with it, so the relative imports
resolve here. `sbsi/lagrangian_score.py` does stay in the package: it is a tested module
(`tests/test_lagrangian_score.py`) and §5C remains the documented Lagrangian route.

Science impact of the §5C failure: **none**. Gold-v1 (+0.245%) and the fiducial Gold-V2
(−0.123 ± 0.152%) both use the §5A transport route and never evaluate a score.

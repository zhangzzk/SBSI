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

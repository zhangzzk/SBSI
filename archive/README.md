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

**Not archived, deliberately.** `../scripts/closure_v2_lagrangian.py` stays in the active tree: it
is the §5C driver, but it also holds the shared V2 plumbing (`rebuild`, `load_rows`,
`scene_context`, `phi_block`) that any V2-side §5B work needs, and the archived probes import it
from there. `sbs_shear/lagrangian_score.py` stays too — it is a tested module
(`tests/test_lagrangian_score.py`) and §5C remains the documented Lagrangian route.

Science impact of the §5C failure: **none**. Gold-v1 (+0.245%) and the fiducial Gold-V2
(−0.123 ± 0.152%) both use the §5A transport route and never evaluate a score.

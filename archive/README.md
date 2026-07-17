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

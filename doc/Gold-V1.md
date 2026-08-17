# Gold-v1 — the certified SBSI shear-calibration pipeline

**Status:** CERTIFIED baseline. Confirmed 2026-07-23 (16-seed ensemble, fixed catalogue),
consistent with `plotting/plot_flow_figures.py` / `figures/`.

**Headline result:** overall multiplicative shear bias **m = +0.245%** on the held-out
constant-gold test set — parameter-free (no fitted correction):

    m = R_sim / (R_flow + R_blend) − 1

where R_sim is the simulation truth response, R_flow the flow's self (shape) response,
R_blend the neighbour-blend response. Bare flow (no blend) is m ≈ +55%; the R_blend
emulator collapses it to sub-percent.

---

## 1. Model

Two pieces, added linearly. **Keep both UNTOUCHED when extending** (e.g. for selection).

### 1a. Measurement flow → R_flow (the shape / self-response)
- **Checkpoints:** `models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt … _s516.pt` (16 seeds).
- **Density:** `P(measured ê₁, ê₂ | conditions)` — conditional affine flow (`mean_affine`),
  `target_dim=2` (`measured_ngmix_g1/g2`), `context_dim=16`, `hidden_dim=256`, `n_layers=3`,
  `n_flows=10`, `mean_hidden=128`, `activation=silu`, `response_difference=central`.
- **Conditions (8):** `[e1_input_p, e2_input_p, sersic_n_input_p, measured_mag_auto,
  measured_flux_radius, nbr_flux_near, nbr_flux_far, nbr_flux_max]` — i.e. TRUE shape + TRUE
  sérsic + **MEASURED mag & size** + **blending flux** (the near/far/max neighbour-flux features).
  Measured mag/size are INPUTS here (this is the measured-conditioned model; NOT the true-property
  reframe, which was a 2026-07 detour — see `memory/project_reframe_outcome`).
- **`flow_drop_indices=[0,1,8,9]`:** the density is BLIND to `e1_input_p, e2_input_p` (indices 0,1)
  and their `__is_missing` flags (8,9). The explicit **mean head** `mu(context)` sees the true shape
  and carries the (un-shrunk) shear response — the density can't, because ML shrinks a first-moment
  derivative. R_flow = the mean-head response.
- **R_flow harvest:** shear the primary's true shape ±g, difference the mean-head projection, with
  **Common Random Numbers** (`--flow-seed 12345`, same latents both legs → sampling noise cancels).
  Per-seed R_flow ≈ 0.29 with **±1% per-seed m scatter** (SGD/cuDNN training stochasticity, NOT
  feature choice). ENSEMBLE R_flow over the 16 seeds → SEM ~±0.1% → the certified m.

### 1b. R_blend — the neighbour-blend response (separate BlendEMU emulator)
- **Emulator tag `lsst_r_extnbr_ho`** (extended-neighbour, held-out; `blendemu/models/`), NOT the old `lsst_r`.
- Per-(case, input_index) response summed over neighbours → lookup `results/blend_lookup_extnbrho_c40-139.feather`
  (symlink to `sbsi_caches/…`), built by `scripts/build_blend_lookup.py --tag lsst_r_extnbr_ho`.
- **Seed-independent**, population-weighted **R_blend ≈ 0.1593** on the held-out set.
- Absent (case,input_index) → R_blend = 0 (correct for isolated / r>26).

---

## 2. Training (`scripts/train_measurement_model.py`, driver `jobs/job_pilot_train.sh`)

- **Catalogue (half-shear, g=0):** `sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather`
  (crowd+conc det_meas, retains undetected rows; firewall — constgold never trained on).
- **feature-set:** `g0_meas_crowd_conc_szfl_noz`; **target-features:** `measured_ngmix_g1 measured_ngmix_g2`;
  `--target-column detected --selection-name sextractor_detected`.
- **flow:** `--flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p`
  `--hidden-dim 256 --condition-layers 3 --n-flows 10`.
- **optim:** `--shear-case 0.0 --epochs 80 --batch-size 8192 --lr 0.0007 --patience 10 --weight-decay 1e-5 --gpu-resident`.
- **response loss:** `--response-weight 450 --response-delta 0.02 --response-difference central`
  `--response-target-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz`
  (the firewall-clean half-shear SNC self-response target; pins the mean-head R_flow to matched-pair truth).
- **16 seeds** `501…516` (ensembled). (SWA variants `..._swaavg_s501-504` also exist.)

---

## 3. Data (validation / acceptance)

- **Constgold catalogue — USE:** `lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather`
  with **`--min-case 40`** (held-out cases 40–139). Fixed-centroid build `const_s4_c2fix` (07-15) →
  **R_sim = 0.4534**.
- **⚠ DO NOT USE** the `constant_response_catalogue_{c40-139,c40-79,c80-139}.feather` splits —
  old reverted-centroid (R_sim ≈ 0.4655, ~+2.7% high, inflates m by ~+3%). **Quarantined 2026-07-23 →
  `*.OLD_do_not_use`.** `job_build_100.sh` rebuilds the bad c40-139 from the old halves (now disarmed).
  See `memory/reference_constgold_catalogue`.
- **Lookups (all match 100% on the held-out set):**
  - `results/blend_lookup_extnbrho_c40-139.feather` — R_blend (BlendEMU `lsst_r_extnbr_ho`).
  - `results/meas_prim_lookup_c0-139.feather` — MEASURED primary observables for constgold
    (`measured_mag_auto`, `measured_flux_radius`, …); constgold has no measured columns natively, so the
    measured-conditioned flow REQUIRES this lookup at validation.
  - `results/crowd_flux_conc_c0-199.feather` — `nbr_flux_near/far/max` blending features.
  - `results/ood_split_c40-139.feather`, `results/blend_multiplicity_extnbrho_c40-79.feather`.

---

## 4. Validation (`scripts/validate_constant_with_blend.py`, driver `jobs/job_pilot_harvest.sh`)

Per seed: `--measurement-model <ckpt> --catalogue …train.feather --min-case 40 --blend-lookup … --crowd-flux-lookup …
--meas-prim-lookup … --ood-lookup … --mult-lookup … --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384`.
Prints `GLOBAL: R_sim=… R_flow(self)=… R_blend(emulator)=…`. Aggregate: average R_flow over seeds
(R_sim, R_blend are seed-independent), then `m = R_sim/(<R_flow>+R_blend) − 1`.

---

## 5. Results (16-seed, 2026-07-23; jobs 15204811 s501-04, 15204921 s505-10, 15204922 s511-16)

- **R_sim = 0.4534**, **R_blend = 0.1593** (single-valued across seeds).
- **⟨R_flow⟩ = 0.2930 ± 0.0032** (SEM 0.0008 over 16 seeds).
- Per-seed m (%): 501 +0.91, 502 −0.70, 503 −0.18, 504 +0.62, 505 +0.38, 506 +1.12, 507 +0.38,
  508 +1.48, 509 +1.09, 510 −0.18, 511 +0.24, 512 −0.07, 513 −0.85, 514 +0.85, 515 −0.24, 516 −0.85
  (spread ±1%, training stochasticity).
- **16-seed ENSEMBLE m = +0.245%.** Bare-flow m ≈ +55% → R_blend → +0.245%.
- Consistent with `plotting/plot_flow_figures.py` (hardcodes `R_SIM=0.4534`, `R_BLEND_TRUE=0.1593`)
  and `figures/fig1..5`.

---

## 6. Caveats & scope

- **Parameter-free / non-circular.** No fitted deficit, no bolt-on correction. This is the honest number.
- **Population:** the DETECTED, quality-selected (`sextractor_detected` + `source_select_selection`) sample.
  m = +0.245% is the *global* bias of that sample.
- **NOT yet covered (→ the next extension):** an ANALYST'S additional MEASURED cut (S/N, measured size/mag)
  moves the selection boundary with shear → a Term-2 selection bias (Sheldon–Huff). Gold-v1 has no explicit
  selection-response term. Extending for that (keeping §1 untouched) is future work — to be done properly on
  the **triad** framework (`scripts/eval_joint_triad.py`: shape / selection / detection), not as a bolt-on.
- **Footgun:** always the fixed `train.feather --min-case 40`; never the quarantined `c40-*` splits.

## 7. Reproduce
```
# validate one seed (repeat s501..s516, average R_flow):
SEEDS="501" TAG=meas_szfl_noz_lam450_fixresp MAXROWS=45000000 sbatch jobs/job_pilot_harvest.sh
# figures:  python plotting/plot_flow_figures.py   (-> figures/fig1..5)
```

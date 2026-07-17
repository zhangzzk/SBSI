# SBSI Pipeline Map

Map of the SBSI shear-calibration pipeline, from **response predictions** to the two
inference directions: **(a) full-Bayesian harvest/inference** and **(b) prob_blending**
(probabilistic-neighbour forward model). Reconstructed 2026-07-16 from `scripts/`, `jobs/`,
the module import graph, and the top of `WORKLOG.md` (newest-first). Newest-first WORKLOG
context is authoritative where this doc and code disagree.

Convention: **[P]** = protected live-pipeline file (do not touch). **[LIVE]** = current
active entrypoint. **[OLDER]** = earlier entrypoint of a live stage, superseded but kept.

---

## 0. Upstream (out of scope — `blendemu/`)
Catalogue building, the XGBoost pair emulator (`f_reg`) and its retrain live in `blendemu/`.
SBSI consumes finished `det_meas_ngmix_*` feathers from
`/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/`. The emulator-retrain jobs
(`job_retrain_ho.sh`, `job_retrain_cw.sh`, `job_retrain_extnbr*.sh`, `job_emu_reweight.sh`,
`job_extnbr_perdist.sh`) `cd blendemu` and run `blendemu/scripts/retrain_extnbr.py` /
`emulator_mean_residual.py` / `emulator_gold_reweight.py` — **these jobs are LIVE** (part of
the cont.44 rebuild chain) even though their target scripts are not under `SBSI/scripts/`.

## Core library — `sbs_shear/`
`measurement_model.py` [P], `spline_flow.py` [P], `scene_model.py` [P], `selection_model.py`,
`posterior_shape.py`, `preprocessing.py`, `shear_map.py`, `coordinates.py`, `sim_stream.py`,
`detection_classifier.py`. Imported by nearly every script (`load_measurement_model`,
`source_select_selection`, `DEFAULT_SELECTION_CUTS`, `apply_shear_to_ellipticity`, …).

`sbs_shear/response.py` is the load-bearing response library, exporting `model_mean_proj`,
`_shape_target_indices`, `load_sheared_sample`, and the `flow_response` (±g secant) helper.
It was promoted (2026-07-17) out of the misnamed `scripts/response_ratio_diagnostic.py`, which
is now a thin **back-compat shim** re-exporting the same objects (byte-identical) plus its
original CLI `main()`, so `from scripts.response_ratio_diagnostic import model_mean_proj` still
works. Imported by `validate_constant_with_blend` [P], `validate_constant_response`,
`validate_allpairs_response`, `eval_self_response_bins`.

---

## DAG

```
 blendemu det_meas_ngmix feathers ($DATA_DIR/sbsi_catalogues)
        │
  [S1] build_detection_measurement_catalogue.py ──► det_meas_ngmix_{ap7,np7,allpairs,...}
        │            (job_build_np7 / _ngmix / _allpairs_* / _g002_100 / _np_g002)
        ▼
  [S1] augment_crowding.py ──► + crowd_flux features
        │  (job_augment_crowd*, job_crowd_conc_prep, job_finalize_ext, job_resp_rebuild_c2fix)
        │
        ├─────────────► [S2] LOOKUPS (shear-independent conditioning tables) ──────────────┐
        │   build_g0_lookup.py            → g0_lookup_c*          (SNC self-response)        │
        │   build_crowding_lookup.py      → crowd_flux(_conc)_c*                             │
        │   build_crowding_lookup_det.py [P] → crowd_flux_det_c*  (detection-context)        │
        │   build_meas_prim_lookup.py     → meas_prim_lookup_c*                              │
        │   build_nn_distance_lookup.py   → nn_dist_c*                                       │
        │   build_ood_lookup / _split     → ood_split_c*                                     │
        │   build_mu_correction.py        → mu correction (infer path)                       │
        │                                                                                    │
        ├─► [S3] TARGETS (flow supervision)                                                  │
        │   compute_response_target_blend.py  → response_target_crowd_rblend_snc_*.npz  ◄────┤
        │       (job_resp_target_*, job_np7_target, job_resp_rebuild_c2fix)   [g0_lookup dep]│
        │   compute_response_target_constant.py → response_target_const_coh_*.npz            │
        │   compute_selection_target_blend.py   → selection_target_*.npz                     │
        │   compare_response_targets.py (RUN_g02tgt Stage A linearity check)                 │
        │                                                                                    │
        └─► [S4] R_blend via BlendEMU emulator                                               │
            build_blend_lookup.py       → blend_lookup_extnbrho_c*   (R_blend per object)    │
            build_blend_multiplicity.py → blend_multiplicity_c*                              │
                                                                                             │
  [S5] TRAIN the measurement flow (self-response model)  ◄── response_target npz ────────────┘
       train_measurement_model.py [P]   → models/measurement_flow_g0_ngmix_*.pt
         [LIVE] job_pilot_train.sh [P]  (tag meas_szfl_noz_lam450_fixresp)
         [OLDER] job_train_meas_szfl / _full / _r2 / _freeze, job_train_conc_g02tgt,
                 job_train_crowd*, job_train_ap7, job_np7_train, job_ensemble_seed/quad
       train_selection_response.py      → selection response-aware model (job_sel_resp_ngmix)
        │
        ├──────────────────► (a) FULL-BAYESIAN ────────────────────┐
        │                                                          │
        │  [S6a] HARVEST / constant-shear m                        │  [S6b] POSTERIOR INFERENCE
        │  validate_constant_with_blend.py [P]                     │  infer_posterior_shape.py
        │    m from R_sim + R_flow + R_blend                       │    → etilde_gold_c40-139.feather
        │    [LIVE] job_pilot_harvest.sh [P]                       │    (job_infer_etilde*, job_crowd_flux_det)
        │      (CAT=constant_response_catalogue_train, MINCASE=40, │  posterior_shape.py (sbs_shear)
        │       --flow-seed CRN)                                   │  flow_response_by_mag.py (R_flow(mag) diag)
        │    uses response_ratio_diagnostic.model_mean_proj [P]    │  measure_flow_c.py / measure_gold_c.py (additive c)
        │    [OLDER] job_constgold_*, job_validate_{100,conc,meas},│
        │      job_harvest_rflow*, job_qdiag_*, job_ood_split      │
        │  validate_allpairs_response.py  (self-response R_sim;    │
        │      ap7/np7/crowd; job_*_ap7, job_validate_crowd*, ...) │
        │  validate_constant_response.py  (pre-blend constant m;   │
        │      job_const_validate_full, job_constgold_lam300)      │
        │  plotting/plot_flow_figures.py → figures/ (5 PNGs;       │
        │      job_fig2_dump, job_fig5_selfresp)                   │
        │                                                          │
        └──────────────────► (b) PROB_BLENDING ◄───────────────────┘
           forward-model the undetected-neighbour R_blend (see PROB_BLENDING.md)
           probblend_characterize.py [P] → probblend_char.feather (det/undet split; job_probblend_char)
           probblend_forward.py [P]      (conditioned production forward model; job_probblend_fwd)
           probblend_calib.py / _calib2d.py  (isotonic classifier recalibration; job_calib, job_calib2d)
           probblend_ctx_diag.py / _gap_diag.py  (per-θ context resolution; job_ctx_diag, job_gap_diag)
           build_crowding_lookup_det.py [P]  (detection-context lookup feeding the forward model)
```

### Additive-bias correction sub-branch (diagnostic/experimental, feeds S6)
`diagnose_additive_origin.py` (shared lib: `attach_lookups`, `flow_mean`, `load_g0`, `RK`) →
consumed by `fit_additive_correction.py` (job_fit_additive_correction),
`finetune_additive_mean_head.py` (job_finetune_additive_mean_head),
`apply_g0_mean_bias_shift.py` (job_apply_g0_mean_bias_shift).
`calibrate_blend_residual_split.py` (job_calibrate_blend_residual_split) reuses
`validate_constant_with_blend.{CBASE,load}` + `response_ratio_diagnostic.model_mean_proj`.

### Scene-model branch (ablation/exploratory, still in scripts/)
`scene_coherent_model.py` (job_scene_coherent), `scene_coherent_field.py`
(job_scene_coherent_field), `scene_ablation.py` (job_scene_ablation). Not on the critical path
to current m; kept as explicit ablations per AGENTS.md. **UNCERTAIN** — candidate for archive
if the coherent-field investigation is closed (confirm with owner).

### Audit / bookkeeping
`audit_blend_truth.py` (job_audit_blend*), `audit_self_truth.py` (job_audit_self*),
`map_truth_cases.py` (job_map_truth), `match_fixed_sample.py` (job_match_fixed),
`ood_rsim_check.py` (job_ood_rsim).

---

## Model & training specification

The certified result is a **response decomposition**, not one end-to-end network:
`m = R_sim / (R_flow + R_blend) − 1`. Three ingredients, produced by different machinery.
Only R_flow is a trained neural model.

### The trained model — R_flow (self / isolated-galaxy shape response)

| Aspect | Detail |
|---|---|
| **Model** | `ConditionalMeanFlow` — explicit-conditional-mean normalizing flow: `p(x\|c) = p_resid(x − μ(c) \| c)`. Defined in `sbs_shear/measurement_model.py` [P]; trained by `scripts/train_measurement_model.py` [P] via `job_pilot_train.sh` [P]. |
| **What it learns** | The selected-object measurement likelihood `p(ê \| true props, neighbour props, shear, detected)` at **g=0** — the isolated/self shape response of a galaxy to shear. |
| **Output / target** | 2-D `(measured_e1_image, measured_e2_image)` — orientation-full measured ellipticity in the spin-2 basis. `target_dim=2`, target-standardized (`TargetStandardizer`). |
| **Architecture** | `--flow-type mean_affine`: a **mean head** μ(c) (MLP, `--mean-hidden 128`) carrying the conditional-mean response, plus a **residual affine-coupling flow** (`ConditionalAffineFlow`: `--n-flows 10`, `--hidden-dim 256`, `--condition-layers 3`, alternating coupling masks, tanh-bounded log-scale, zero-init last layer). Residual flow is **blind** to the shape features (`--flow-blind-features e1_input_p e2_input_p`) so the shape→shape response is forced into the mean head and cannot be shrunk by max-likelihood. Optimizer Adam, `--lr 0.0007`, `--patience 10`, up to 80 epochs. |
| **Conditioning (context)** | `DEFAULT_MEASUREMENT_CONDITION_FEATURES` = `DEFAULT_SELECTION_FEATURES` + `redshift_input_p` + `redshift_input_s_blend`. Primary: `Re_input_p_scaled`, `r_input_p_scaled`, `sersic_n_input_p`, `e_abs_p`, `gamma_pframe_{parallel,cross}_p`. Neighbour-gated secondary/pair (all `_blend`-gated by `neighbored`): `distance_scaled_blend`, `flux_ratio_blend`, `Re_input_s_scaled_blend`, `r_input_s_scaled_blend`, `sersic_n_input_s_blend`, `pair_pframe_{cos2,sin2}_blend`, `e_pframe_{parallel,cross}_s_blend`, `gamma_pframe_{parallel,cross}_s_blend`. ~20 features, spin-2 canonical basis `(q1,q2)=q(cos2θ,sin2θ)`. |
| **Loss** | `L = L_NLL + λ · L_resp` (`epoch_response` in `train_measurement_model.py`). `L_NLL` = mean negative log-likelihood, optionally decorrelation-weighted over \|e\|×size bins (`compute_decorrelation_weights`). `L_resp` = **fractional** response penalty `Σ_bins (R_model/R_sim − 1)²` over true-flux×size bins — penalizes the per-bin multiplicative bias itself, not absolute `(R_model−R_sim)²`. `R_model(bin)` via a **central** finite-shift secant (`--response-difference central`, `--response-delta 0.02`) through the analytic shear map `S_δ` applied to the intrinsic shape. `λ = 450` (`--response-weight`, tag `meas_szfl_noz_lam450_fixresp`). `--weight-decay 1e-5`. |
| **Response supervision** | `R_sim(bin)` from `results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz` (6×3×5 flux/size bins), built by `compute_response_target_blend.py` [S3]. |
| **Training data** | `$DATA_DIR/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather` (blendemu g=0 detection-measurement catalogue). ~23.7M selected rows after cuts. |
| **Train/val split** | `sklearn.train_test_split`, `random_state=<seed>` (`split_data`). ≈23.66M train / 4.18M val — the `nll=train/val` pair in the training logs. |
| **Selection cuts** | `source_select_detection` (`DEFAULT_CUTS`): `r_input_p∈[18,28]`, `Re_input_p∈[0.1,1.5]`, and (`distance∈[0,5]` **or** not-neighboured). SExtractor detection defines `s=1`. |
| **Seeds / ensemble** | Seeds 501–508 trained; 509–516 queued. Only R_flow varies across seeds (R_sim, R_blend seed-independent). CRN harvest (`--flow-seed 12345`, same seed for both ±g legs) makes R_flow deterministic + position-independent. 8-seed ensemble mean R_flow = 0.2919, σ = 0.0032. |

### The two non-neural ingredients

| Ingredient | What / how | Value (fixed convention) |
|---|---|---|
| **R_sim** | Truth response measured directly on the **constant-shear** sims (`constant_response_catalogue_train.feather`, `--min-case 40`), fixed-centroid convention. A measurement, not a model. | **0.4534** |
| **R_blend (true neighbours)** | **BlendEMU** XGBoost pair emulator (`f_reg`, lives in `blendemu/`) evaluated on each detection's **actual** neighbours; per-object lookup `blend_lookup_extnbrho_c*`. Additive blend contribution to the response. | **0.1593** |
| **R_blend (probabilistic neighbours)** | `probblend_forward.py` [P] — neighbours **drawn from the population prior** (Poisson / `Φ×[1−p_det]` forward model, no true neighbours), then emulated; `probblend_char.feather`. | (see PROB_BLENDING.md) |

### Certified result

`m = R_sim / (R_flow + R_blend) − 1 = 0.4534 / (0.2930 + 0.1593) − 1 = **+0.245% ± 0.268%**`
(16-seed CRN ensemble, flow-seed 12345), fully parameter-free (no empirical scalars) — consistent
with zero at 0.9σ, subpercent and under the ≤0.3% aspiration. The m=0 point is
R_flow = R_sim − R_blend = 0.2941; the residual +0.245% is R_flow sitting 0.0011 below it. Error
budget (WORKLOG cont.50–51): σ_m = 0.182% (seed, ∝1/√N) ⊕ 0.195% (R_sim finite-case floor,
irreducible by more seeds) ⊕ 0.017% (R_blend). The earlier N=8 "+0.51%" was a small-N R_flow
fluctuation (0.2919→0.2930 at N=16). See WORKLOG cont.51.

**Known limitation — tomographic non-closure (WORKLOG cont.52).** The global +0.245% is a
*cancellation across magnitude*: per magnitude bin, ⟨R_flow⟩+⟨R_blend⟩ vs ⟨r_sim⟩ ranges from
≈−2.3% (bright) to ≈+8.3% (faint), driven at faint mags by the BlendEMU R_blend term. A
magnitude-tomographic or faint-weighted (LSST-like) analysis would see up to ~8% response
miscalibration even though the global mean is subpercent. (Numbers from the s501 low-outlier seed;
the *structure* is seed-robust, exact magnitudes want a multi-seed per-bin repeat.)

---

## Current LIVE entrypoints (per WORKLOG cont.51–52, 2026-07-17)
- **Train:** `job_pilot_train.sh` → `train_measurement_model.py` (tag `meas_szfl_noz_lam450_fixresp`).
- **Harvest / m:** `job_pilot_harvest.sh` → `validate_constant_with_blend.py`
  (`CAT=constant_response_catalogue_train.feather`, `MINCASE=40`, `--flow-seed` CRN).
- **c2-fix rebuild chain:** `job_resp_rebuild_c2fix.sh` → `job_retrain_ho.sh` (blendemu emulator)
  → `job_build_4079.sh` / `job_build_100.sh` → `job_blend_multiplicity_4079.sh` → re-harvest.
- **Figures:** `plotting/plot_flow_figures.py` → `figures/` (5 PNGs; per-object dumps via
  `job_fig2_dump.sh`, `job_fig5_selfresp.sh` → `eval_self_response_bins.py`).
- **Posterior inference:** `job_infer_etilde*.sh` → `infer_posterior_shape.py`.
- **prob_blending:** `probblend_forward.py` / `probblend_characterize.py` (+ calib).
- **Closed experiment (archived):** the g=0.02-response-target check (RUN_g02tgt) ran Stage A
  2026-07-12 → verdict LINEAR, Stage B never launched; runbook + `compare_response_targets.py`
  moved to `archive/`. See WORKLOG cont.16–18.

## Superseded-but-kept entrypoints (kept in `scripts/`)
- `validate_constant_response.py` — pre-blend constant validator; superseded by
  `validate_constant_with_blend.py` but still referenced by `job_const_validate_full.sh`,
  `job_constgold_lam300.sh`; also a de-dup source for the shared response lib.
- `validate_allpairs_response.py` — ap7/np7-era self-response validator; still used by the
  crowd-era validate jobs, so kept.

---

## Repository layout (post-2026-07-17 cleanup)

```
sbs_shear/          core library (installable: pip install -e .)
  measurement_model.py [P]  spline_flow.py [P]  scene_model.py [P]
  selection_model.py  detection_classifier.py  posterior_shape.py
  preprocessing.py  shear_map.py  coordinates.py  sim_stream.py
  response.py               R_flow response lib (model_mean_proj / flow_response / …)
scripts/            29 live-pipeline + kept-older scripts (build / target / train / harvest / infer / probblend)
plotting/           plot_flow_figures.py  (the 5 certification figures)
jobs/               100 Slurm jobs for the above (+ jobs/archive/ for superseded)
archive/            superseded scripts + closed one-off investigations (git-reversible, still runnable via SBSI_ROOT)
tests/              pytest unit tests (run in the py31 env: python -m pytest tests/)
results/, models/   generated artifacts — gitignored; large keepers live under $DATA_DIR/sbsi_caches (symlinked)
```

## Public API (`sbs_shear`)

Future jobs/scripts should import from the library rather than re-inlining logic:

```python
from sbs_shear import load_measurement_model          # load the trained R_flow flow bundle
from sbs_shear.response import flow_response           # R_flow = (m_+g − m_−g)/(2g), optional per-object + CRN reseed
from sbs_shear.response import model_mean_proj, load_sheared_sample
from sbs_shear.preprocessing import source_select_selection, DEFAULT_SELECTION_CUTS
```

- **Train** (R_flow): `job_pilot_train.sh` → `train_measurement_model.py`
  (tag `meas_szfl_noz_lam450_fixresp`, `--response-target-npz results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz`).
- **Harvest / m:** `job_pilot_harvest.sh` → `validate_constant_with_blend.py`
  (`CAT=constant_response_catalogue_train.feather`, `MINCASE=40`, `--flow-seed 12345`).
- **Infer:** `job_infer_etilde*.sh` → `infer_posterior_shape.py`.

## Background / how we got here (condensed; full history in WORKLOG.md, newest-first)

Shape estimator since 2026-07-01 is **ngmix** `NGMIX_G1/G2` (PSF-corrected reduced shear); earlier
SExtractor windowed-moment numbers are PSF-diluted and kept only for the record.

1. **Baseline & closure.** Flow-MLE recovery m=+3%; closure test unbiased (<0.2%) → machinery sound.
2. **Bias = shear-response miscalibration, not g=0 fidelity.** Seven converging diagnostics → m is a
   response-transfer gap, decoupled from g=0 likelihood quality.
3. **Direct response calibration → sub-percent (SExtractor era).** First-moment (BFD-like) estimator,
   responsivity from g=0.05, tested on independent g=0.02 → m=−0.24%±0.63% (28M rows).
4. **Response-aware flow training.** Sobolev loss `L = NLL + λ‖R_model − R_sim‖²` + MLP mean head +
   property-resolved (flux×size×blend) supervision → the flow's induced response tracks R_sim.
5. **Gold constant-shear validation** revealed the flow target and gold measured *different estimators*
   → triggered the ngmix pivot.
6. **Estimator pivot to ngmix** (2026-07-01).
7. **The 0.30-vs-0.39 isolated-response gap = coherent blending** (not a bug): coherent-field shear
   boosts the ngmix response via aligned neighbour light. **Split of labor** — the SBSI flow models the
   incoherent self-response (R_flow); the external **BlendEMU** emulator supplies the coherent blend
   part (R_blend); combined they reproduce constant-shear (→ the decomposition above).

## Refactor status (2026-07-17)

Executed from the former `MODULARIZATION_PLAN.md` (now folded here): response library promoted to
`sbs_shear/response.py` + shim; `pyproject.toml` added; closed branches (scene / additive / superseded
selection) + one-off diagnostics archived; markdown set consolidated. **Not done** (deliberately, to keep
the certified path byte-identical — see the "freeze core" decision): the deeper `sbs_shear/io.py` +
`lookups.py` extraction that would rewire the protected harvester/trainer, and the physical regroup of
`scripts/`/`jobs/` into stage subdirs. Those remain available if a future pass re-verifies m end-to-end.

## Cleanup history
- **2026-07-17** — this cleanup: git-initialised the repo; `sbs_shear/response.py` extraction; 19 scripts
  + 57 jobs archived; superseded outputs deleted / live keepers relocated to `$DATA_DIR`; docs consolidated
  (`MODULARIZATION_PLAN.md`, `SUMMARY.md` folded here; `RUN_g02tgt.md` archived). See WORKLOG cont.53.
- **2026-07-16** — toy-investigation cluster (7 `toy_*`), `measure_flow_c_train.py`, `neighbor_shear_null.py`
  + their jobs archived (recoverable from git history / `archive/`).

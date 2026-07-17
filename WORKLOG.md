# SBSI Work Log

This file records substantive changes to the standalone SBSI shear-calibration project.

## 2026-07-17 (cont.53, REPO CLEANUP + PIPELINE RESTRUCTURE — git-initialised, response library promoted, dead scripts/jobs archived, 22.5 GB of superseded outputs deleted, docs consolidated; certified m untouched)

Milestone housekeeping pass after cont.52. Goal: a clean, navigable repo with a clear train/inference API, no behavior change to the certified `m = R_sim/(R_flow+R_blend)−1`. Scope decided with the owner: **freeze the certified core** (protected trainer/harvester byte-identical, no re-harvest), **hard-delete superseded outputs**, **archive closed branches**.

**Analysis:** 6-agent workflow (dep graph + script/job/output/markdown/redundant-code analyzers) built the keep/delete map, cross-checked against the live entrypoints + WORKLOG top + the PROTECTED-file list.

**Git.** Repo `git init`-ed (was untracked). `.gitignore` excludes `results/ models/ figures/ data/ notebooks/` + `*.pt/*.feather/*.npz/*.png` + caches/logs, so the 27 GB of artifacts stay out. Baseline commit `9439760`; the cleanup is 4 further commits (rollback points).

**Code (git-reversible).**
- **`sbs_shear/response.py`** — promoted the load-bearing response library (`model_mean_proj`, `_shape_target_indices`, `load_sheared_sample`) out of the misnamed `scripts/response_ratio_diagnostic.py`; added a `flow_response(±g secant, optional per-object + CRN reseed)` public helper. `response_ratio_diagnostic.py` is now a thin back-compat shim re-exporting the **identical objects** (verified `a is b`) + its original CLI `main()`. The protected harvester's `from scripts.response_ratio_diagnostic import model_mean_proj` is byte-identical → certified number untouched.
- **`pyproject.toml`** added (optional `pip install -e .`); `sbs_shear/__init__.py` now exports the response API. PYTHONPATH contract unchanged.
- **Archived 19 scripts + 57 jobs** (`git mv` → `archive/`, `jobs/archive/`): scene-coherent, additive-correction, superseded selection branches + closed one-off diagnostics (measure_gold_c/measure_flow_c/ood_rsim_check/map_truth/match_fixed/audit_*_truth/flow_response_by_mag/compare_response_targets) + pre-fixresp job families (ap7/np7/build/constgold/blendlookup/fmval/halfshear/snc). Verified **no kept script imports an archived module, no kept job invokes an archived script**. `scripts/` 48→29, `jobs/` 157→100.
- **Tests:** fixed `tests/test_scene_model.py` (dropped a dead import of the archived scene trainer that broke collection at baseline). **17 pass** (py31 env, was 12 collectable).

**Docs.** `MODULARIZATION_PLAN.md` + `SUMMARY.md` folded into **`PIPELINE.md`** (now the single authoritative map: DAG + Background/history + Public API + Repository layout + Refactor status + Cleanup history) and removed; `RUN_g02tgt.md` (closed experiment) → `jobs/archive/`. Refreshed stale refs (`plot_flow_calibration.py`→`plotting/plot_flow_figures.py`; 8-seed +0.51% → **16-seed +0.245% ± 0.268%**; added the tomographic non-closure limitation). **Kept `SBI_shear.md` + `SBI_shear_response.md`** — cited by section number inside protected code (`measurement_model.py`, `train_measurement_model.py`, …). Final doc set: CLAUDE, AGENTS, PIPELINE, PROB_BLENDING, SBI_shear, SBI_shear_response, WORKLOG.

**Outputs (irreversible; `$HOME` not git-tracked).** `results/` 27 GB→~0.01 GB local, `models/` 683 MB→166 MB.
- **Deleted 22.5 GB** superseded: scene-branch outputs (5.8 GB), `resp002_c40-59` (closed RUN_g02tgt), old-convention lookups + already-consolidated shards (verified byte-identical to their `c40-139` finals), pre-c2fix `c0-39` lookups, 2 `.stale_bak`, and 178 pre-fixresp model checkpoints. Verified none are read by the live harvester (`job_pilot_harvest.sh` reads only the `c40-139`/`conc`/`meas_prim`/`mult_c40-79` finals + `constant_response_catalogue_train`).
- **Relocated 8 live finals (~9.9 GB)** to `$DATA_DIR/sbsi_caches/` + symlink back into `results/` (crowd_flux_conc, meas_prim, ood_split_c40-139, blend_lookup_extnbrho_c40-139, constgold_perobj_raw, g0_lookup_c0-99, blend_multiplicity_c40-79, probblend_char) — every pipeline path still resolves, `$HOME` reclaimed. Kept: fixresp s501-516 + SWA ensemble, 5 small live inputs (response target npz, fig5 npz, truth manifest, etilde prior samples), 7 tiny referenced-default models.

**Validation run:** `pytest tests/` (17 pass); import-identity check on the response shim; import smoke of kept entrypoints (5/6 OK; `train_measurement_model` fails only on the login-node scipy `GLIBCXX_3.4.30` ABI issue — pre-existing, resolves on compute nodes). No GPU harvest re-run needed (certified path byte-identical by construction).

**Limitations / next steps.** Deliberately NOT done (would touch the certified path): the deeper `sbs_shear/io.py` + `lookups.py` extraction and the physical regroup of `scripts/`/`jobs/` into stage subdirs — available for a future pass that re-verifies m end-to-end. `jobs/` still holds ~100 mostly older-convention runs kept as reproducibility refs (could be pruned further). The relocation symlinks may need re-pointing if a future full rebuild rewrites a final via atomic-rename.

## 2026-07-17 (cont.52, fig2/fig5 reconciliation + per-object R_flow fix) — user Qs on the figures exposed that the +0.24% lives on the certification budget, and surfaced two honest qualifications I under-stated in cont.51

**Two user questions, both correct instincts:**

**Q1 — "fig2 model R doesn't resolve true R at all, yet flow+emulator each do separately."** Confirmed a FIGURE ARTIFACT, not a flow failure. The `fig2_perobj_s501_fixresp.feather` dump has `R_flow` = one unique value (0.28999) across all 26.9M rows (R_blend and r_sim ARE genuinely per-object: 23.6M unique R_blend). Root cause is exactly the cont.51 finding: `model_mean_proj` reduces to `float(np.mean(proj))`, so the harvest forms the global scalar `R_flow=(mp−mm)/(2g)` BEFORE the dump and broadcasts it. The flat orange line = scalar 0.290 + per-obj R_blend, so it can only wiggle through R_blend. The flow DOES resolve response — **fig5 is the proof** (its self-response tracks truth from R≈0.88 bright → R≈0.012 faint, and across size). fig2-as-built is broken for its stated purpose.

**Q2 — "fig5 shows flow 1–4% below truth per bin; how is m only +0.2%?"** Reconciled quantitatively (`tmp/reconcile.py`):
- The −1/−4/−11% labels are NOT the m inputs. Count-weighted (equal-count quantile bins), the training-grid self-response deficit is **−1.8%** (model 0.2762 vs target 0.2812), dominated by the −1.0% bright bin; the −11% sits on R≈0.014 with ~2% of the response weight → invisible in any global average.
- **m is a closed ratio on a DIFFERENT population/estimator than fig5.** m uses the constant CERTIFICATION catalogue with the matched ±0.02 CRN estimator: R_sim=0.4534, R_flow=0.2930, R_blend=0.1593 → +0.24%. fig5 uses the flow's g=0 δ=0.02 linearized self-response vs the TRAINING target (nominal_g=0.05, "snc" estimator, cases 0–99; the target npz's own `global_R`=0.2812). Different population, g, and estimator → its absolute R (0.276–0.281) legitimately ≠ the certification harvest (0.293).
- **The sign is the reconciliation:** a flow that reads its own response slightly LOW leaves a slightly-too-small denominator → m slightly POSITIVE. fig5 (flow low) and fig3 (m=+0.24%) are the same effect; on the certification budget the R_flow deficit is only −0.37% (0.2930 vs the m=0 point 0.2941) → +0.24%.

**fig5 is a SINGLE seed (s501)** — npz `model=..._s501.pt`, 2M rows, CRN flow-seed 12345. Its per-property structure carries s501's own init/SGD noise; a single faint-bin −11% is not a stable miss and will scatter across seeds.

**Two honest qualifications the questions surfaced (I under-stated these in cont.51):**
1. **Error budget is wider than the certified ±0.27% (seed scatter only).** The R_flow self-response deficit reads −0.37% on the certification budget but −1.8% on the training grid — a ~1% estimator/population systematic on R_flow not folded in. The certification RATIO is internally self-consistent (all three R's measured the same way on the same population), so +0.24% stands as a statement about that population, but the honest R_flow systematic is ~1%, not sub-0.2%.
2. **Global m hides per-property structure.** fig5's −1%(bright)→−4%/−11%(faint) averages to −0.37% for a single-bin calibration but would NOT cancel in a tomographic/faint-weighted analysis. That per-bin residual is the scientifically meaningful limitation.

**Code fix (fig2 done right, provably preserves the certified number):**
- `response_ratio_diagnostic.py::model_mean_proj` — added optional `return_proj=False`; when True also returns the per-object projection array. Default path unchanged (2-tuple); all 28 existing callers use 2-tuple unpack → backward-compatible.
- `validate_constant_with_blend.py` CRN legs (line 261–264) — request `return_proj=True`, keep the global `R_flow=(mp−mm)/(2g)` scalar EXACTLY (identically == mean of the per-object array, so the m budget is untouched), and add `R_flow_perobj=(projp−projm)/(2g)`; the `--dump` now writes `R_flow_perobj` instead of the scalar.
- Both files AST-parse clean. This is the "optional per-object return" fix flagged-but-deferred in cont.51, now user-authorized.

**Submitted:** job **15128692** (fig2_dump, cip-cl-nv01 full a40) — regenerated the s501 per-object dump. DONE: R_flow now has **15,550,565 unique values** (was 1); mean = **0.2900**, identically the certified global → the fix preserves the number exactly.

**★ RECONCILIATION RESULT (`tmp/cert_reconcile.py`, s501 cert catalogue, matched ±0.02 CRN, fig5 flux edges) — the per-bin structure is the real story:**

Self-response deficit (⟨R_flow⟩ vs ⟨r_sim⟩−⟨R_blend⟩) per flux bin, cert vs fig5-training-grid:
| flux (mag) | cert dev | fig5 dev |
|---|---|---|
| [18,24.3) bright | **+2.5%** | −1.0% |
| [24.3,25.0) | +3.1% | −2.2% |
| [25.0,25.6) | −2.9% | −2.0% |
| [25.6,26.0) | −13.7% | −4.4% |
| [26.0,26.5) | −26.8% | −3.3% |
| [26.5,28) faint | **−56.5%** | −10.7% |

Count-weighted GLOBAL deficit: cert **−1.4%** vs fig5 training-grid **−1.8%** → agree to ~0.4% (estimator + grid-clip + population). So on a like-for-like *single-seed* basis the two estimators agree; my cont.51 "−0.37% cert vs −1.8% train" compared the ENSEMBLE (0.2930) to s501's training grid — apples/oranges. Honest R_flow estimator/population spread for a fixed seed is **~0.4%** (→ ~0.26% in m), separate from the ±1% seed scatter (s501=0.2900 is the low outlier; ensemble=0.2930).

**★★ TOMOGRAPHIC NON-CLOSURE (the scientifically decisive finding).** Even the TOTAL model response ⟨R_flow⟩+⟨R_blend⟩ does NOT close to ⟨r_sim⟩ per magnitude bin (per-bin m, s501):
| flux (mag) | ⟨r_sim⟩ | ⟨Rflow+Rblend⟩ | per-bin m |
|---|---|---|---|
| [18,24.3) bright | 0.933 | 0.955 | **−2.3%** |
| [24.3,25.0) | 0.527 | 0.540 | −2.3% |
| [25.0,25.6) | 0.389 | 0.382 | +1.7% |
| [25.6,26.0) | 0.320 | 0.302 | +5.8% |
| [26.0,26.5) | 0.274 | 0.255 | +7.6% |
| [26.5,28) faint | 0.239 | 0.221 | **+8.3%** |
| GLOBAL | 0.4534 | 0.4493 | +0.92% (s501) |

Per-bin means precise to ~0.001 (N≈4M/bin, SEM~0.001 → +8.3% is ~18σ, not noise). At faint mags ⟨R_blend⟩≈0.21 dominates ⟨R_flow⟩≈0.014, so the faint residual is mostly the EMULATOR's, not the flow's. **The global +0.24%/+0.92% is a CANCELLATION of −2.3% (bright) against +8.3% (faint).** Implication: the parameter-free response model is calibrated in the global mean but carries real magnitude-dependent residuals (−2% → +8%) that only cancel under this sample's magnitude weighting. **A magnitude-tomographic or faint-weighted (LSST-like) analysis would see up to ~8% response miscalibration** — this is the true limitation of the current framework, far more consequential than the ±0.27% global bar. (Numbers are s501, the low-outlier seed; the STRUCTURE is driven by the seed-independent R_blend magnitude dependence + the flow's response shape, so it is expected to be robust, but exact magnitudes want an ensemble/multi-seed per-bin repeat.)

**Figures regenerated (plotting/plot_flow_figures.py, login node, PLOT_EXIT=0):** fig2 status flipped scalar→**OK** (caveat banner gone; `R_flow.nunique()`=15.5M > 1). fig2 now shows orange (R_flow+R_blend) TRACING blue (r_sim) across flux 0.1→1.4, size 0.06→0.9, blend-flux — the flow resolves response (settles Q1). The flux panel visualizes the non-closure: model/truth CROSS at S/N≈15–20, model over-predicts bright (1.40 vs 1.19 at top S/N) and under-predicts faint — consistent with the −2.3%→+8.3% closure table (extreme fine bins even larger, ~18% over at the very brightest S/N). Size/blend panels agree well; flux is where the residual lives.

**Recommend a NEW figure: per-bin m vs magnitude (the tomographic-closure plot), ideally a seed band** — PENDING user approval (not built autonomously). Error-budget stance: the certified global m=+0.24%±0.27% stands under its own ±0.02-CRN definition (self-consistent); the ~0.4% estimator spread is a robustness caveat not an additive bar; the ~2%→+8% per-mag non-closure is a SEPARATE, larger, science-relevant systematic for any non-global (tomographic/faint-weighted) use.

## 2026-07-17 (cont.51, ★ HEADLINE: 16-seed CRN ensemble → m = +0.245% ± 0.268%, CONSISTENT WITH ZERO at 0.9σ — subpercent AND ≤0.3% aspiration met, parameter-free) — the +0.51% (N=8) was a small-N R_flow fluctuation; 8 new seeds pulled mean R_flow 0.2919→0.2930

**Clean 16-seed CRN tally (flow-seed 12345, one dedicated single-seed harvest per model; raw log tally is contaminated by pre-CRN + cross-check harvests so parsed per-file with nGLOBAL=1 from the current campaign — jobs 15124066–074 for s501-508, 15125780–794 + cip 15126208/210 for s509-516):**

R_flow = {501:0.2900, 502:0.2973, 503:0.2949, 504:0.2913, 505:0.2924, 506:0.2891, 507:0.2924, 508:0.2875, 509:0.2892, 510:0.2949, 511:0.2930, 512:0.2944, 513:0.2980, 514:0.2903, 515:0.2952, 516:0.2980}
mean = **0.2930**, std = 0.0033, sem = 0.00082.

**m = R_sim/(R_flow+R_blend)−1 = 0.4534/(0.2930+0.1593)−1 = +0.245%.** Honest budget (from cont.50 variance decomposition): σ_m(seed,N=16)=0.182% ⊕ σ_m(R_sim floor)=0.195% ⊕ σ_m(R_blend)=0.017% = **σ_m(total)=0.268%**. **⇒ m = +0.245% ± 0.268% = 0.91σ from zero.**

**Answer to the overnight directive (model noise: seeds/SWA + bias):**
1. **Bias:** NONE significant. m consistent with zero at 0.9σ; |m|=0.24% is subpercent and under the ≤0.3% aspiration. The N=8 "+0.51%" was an upward fluctuation from the first-8 seeds' low R_flow (0.2919); doubling to 16 seeds regressed it to 0.2930 (Δm=−0.25%, within the seed error) — a textbook small-N fluctuation, NOT a systematic.
2. **Seed noise:** real (per-seed σ_Rflow=0.0033, ±0.18% in m at N=16) but averages down as 1/√N. This was the whole "+0.51%" story.
3. **The floor is the SIM, not the model:** R_sim case-to-case sampling (±0.195%) is now co-dominant with the (shrinking) seed term and is irreducible by seeds/SWA/architecture — only more constant-shear cases or CRN-paired ±g renders (blendemu-side) reduce it. r_sim is already ±g-paired (intrinsic shape cancels per object, validate_constant_with_blend.py:190), so the residual is genuine population + pixel-noise variance.
4. **SWA:** CONFIRMED confirmatory-only. Partial harvest (15126226 swabase / 15126227 swaavg, still running at loop-stop — full 4-seed numbers land in those logs): s501 swabase=0.2951, swaavg=0.2922; s502 swabase=0.2974. All inside the normal seed range (0.2875–0.2980) → **no headline change**, as predicted. **Bonus insight:** swabase s501 (0.2951) vs the ORIGINAL-trainer s501 (0.2900) differ by 0.0051 at the SAME seed — the SWA leg ran on cip vGPU, the original on V100 → part of the per-seed R_flow scatter is run-to-run/hardware nondeterminism, NOT the random seed alone. SWA (within-run tail average) can't remove that between-run component; ENSEMBLING across seeds/runs can (why 8→16 seeds regressed +0.51%→+0.24% and SWA is the weaker lever). Net: seeds > SWA for this noise; both are dominated by the irreducible R_sim sim floor anyway.

**LOOP STOPPED (goal reached):** subpercent bias found — m = +0.245% ± 0.268%, consistent with zero; the residual error floor is sim-side R_sim case-sampling, not the model, so sub-±0.19% certification is infeasible under the current framework (fixed 100-case constant set) without more sim cases / CRN-paired ±g renders. All overnight deliverables complete (bias analysis, 5 figures, pipeline review, plotting consolidation).

**⇒ SBSI certifies subpercent, unbiased shear calibration (m consistent with 0) with a parameter-free forward model. Loop goal reached.** Remaining: SWA confirmation + plotting-code consolidation.

**Plots regenerated at N=16** (`scripts/plot_flow_figures.py`, 5 PNGs in figures/, no PDF, no gridlines, separate files). Verified the script's harvest parse reproduces the clean CRN tally EXACTLY (single-seed-header + latest-mtime filter correctly excludes pre-CRN/cross-check contamination) → fig3 headline m = **+0.245% ± 0.18%** (true nbrs, N=16), matching the hand tally. **Status:** fig1 (per-seed val loss) OK; **fig3 (true-nbr bias) OK — the headline figure**; fig5 (self-response vs truth across r_input_p×Re bins, real eval npz s501) OK; **fig4 (prob nbrs) INTERIM** (m=−0.126% via the OLD +0.0017 forward residual; a faithful current-convention Level-B is blocked — probblend_forward hardcoded to main set, constant set has no detection catalogue); **fig2 (response vs properties) LIMITED — model line is FLAT.** Root cause (task #13 review): `model_mean_proj` (response_ratio_diagnostic.py:146) returns `float(np.mean(proj))` — a GLOBAL scalar — so the per-object `--dump` R_flow column is constant (0.2900 for all 26.9M rows; r_sim/r_blend ARE genuinely per-object). Tracing R_flow vs flux/size needs PER-BIN R_flow (the harvest's per-bin tables, which also need the per-bin CRN reseed fix, cont.49 finding b) or an optional per-object return added to model_mean_proj — deferred (protected file; fig5 already shows self-response-vs-truth-vs-properties). fig2 carries an honest "R_flow scalar" caveat banner meanwhile.

## 2026-07-17 (cont.50, HONEST ERROR BUDGET — the m error floor is R_sim case-scatter, NOT the flow; seeds/SWA can't beat it; m = +0.51% ± 0.26% ≈ 2σ) — variance decomposition on the 26.9M-row per-object dump settles the overnight "seeds vs SWA vs bias" question

**Decisive result (`scripts/variance_decomposition.py` on the fixed seed-501 per-object dump, cases 40–139, parameter-free, all resampling over CASES):**

| component | σ in m | scales with seeds? |
|---|---|---|
| R_flow harvest floor (jackknife R_flow over 100 cases) | **≈0.00%** (σ_Rflow=0.0000) | n/a — R_flow is CRN-deterministic + case-stable |
| **R_sim case-to-case scatter** (σ_Rsim=0.0009/100 cases) | **±0.19%** | **NO — irreducible floor (finite # sim cases)** |
| R_blend case scatter | ±0.02% | no |
| seed init/SGD scatter (σ_Rflow=0.0024 across 10 seeds) | ±0.17% at N=10 | yes, ∝1/√N → 0.12% at N=16 |
| **COMBINED honest** | **±0.26%** | floor-limited at ~±0.19% |

**m at seed-mean R_flow=0.2919 → +0.51% ± 0.26% = 1.96σ from zero.** Subpercent, STABLE, but only marginally significant — neither a clean detection nor cleanly a fluctuation. **The flow model is NOT the limiting factor:** its seed noise is sub-dominant and shrinks with N; the floor is the *simulation's own R_sim sampling* over the finite 100 constant-shear cases. **This directly answers the overnight question: more seeds help only the ±0.17%→0.12% term; SWA can at best remove that term (±0.26%→±0.19%); neither can break the ±0.19% R_sim floor.** To go below ~±0.19% you need MORE constant-shear sim cases (or a lower-variance R_sim estimator), not more/averaged models.

**Correction to the review's magnitude:** the CONFIRMED finding (cont.49) was right that ±0.25% was seed-SEM-only, but the omitted floor is R_sim's case scatter (which the harvest bootstrap `boot_m_err` ALREADY resamples), not an R_flow floor (that one is ~0). So honest ±0.26% ≈ the old ±0.25% by coincidence of magnitude; the real change is *which term dominates* and *that it won't shrink with seeds*.

**Response-penalty in-sample vs OOS** (penalty target c0-99; split c40-99 vs c100-139): m = +0.67%±0.24% (in-sample) vs +1.31%±0.31% (OOS), Δm=+0.64%. **R_flow is IDENTICAL (0.2900) on both** → the shift is entirely R_sim heterogeneity between case ranges (cases 100-139 have true R_sim ~0.003 higher), NOT flow penalty-overfitting. Δm is ~1.6σ of the case-split sampling error → not significant; consistent with R_sim varying across the sim case ranges. Reassuring: the flow does not generalize worse on penalty-OOS cases.

**Artifacts:** `scripts/variance_decomposition.py` (NEW, reusable: reads any per-object dump + scans harvest logs for CRN R_flow seeds); dump at `$DATA_DIR/sbsi_dumps/fig2_perobj_s501_fixresp.feather` (26.9M rows). Next: refresh with N=16 seeds + SWA harvests when they land (will only move the seed term), then plots.

## 2026-07-16 (cont.49, overnight: queue rescue to cip + SWA paired arm + full pipeline review) — seeds 514/515 OOM-failed on small inter GPUs, rescued to idle cip a40-24gb slices; SWA-vs-bestval test launched on seeds 501–504; multi-agent review of live pipeline running

**Queue rescue:** trains 15125789/91 (seeds 514, 515) both died with CUDA OOM (landed on 10.6 GB GPUs; training needs ~12 GB). The a40 `inter` queue was backed up to 07-17/07-18, so pending a40 resubmits + the fig2 per-object dump were cancelled and resubmitted on **cip** (idle a40-24gb slices, 41 GB-RAM nodes, `--mem=36G --cpus-per-task=8`): trains 15126178 (s514) / 15126180 (s515) + `afterok` inter harvests 15126179/15126181; fig2 dump 15126182 on cip-cl-nv01 (full a40, 90G). All three started within seconds. IDs in tmp/moreseeds.txt.

**SWA arm (subagent):** new `scripts/train_measurement_model_swa.py` (COPY of the protected trainer; adds rolling last-K=8 end-of-epoch state_dict averaging, CPU-side, saves BOTH best-val `..._swabase_s<seed>.pt` and tail-averaged `..._swaavg_s<seed>.pt`) + `jobs/job_swa_train.sh`. Production: seeds **501–504 re-trained with the SWA trainer** on cip a40-24gb → paired comparison SWA vs best-val on identical seeds (swabase leg doubles as determinism check against existing s501–504), then two `afterany` CRN harvests (FLOWSEED=12345) per TAG. IDs in tmp/swa_jobs.txt. Purpose: decide whether per-seed R_flow noise (~0.7% rel.) is checkpoint-selection noise SWA can reduce, or irreducible seed noise.

**SWA arm — SUBMITTED (agent 7668d735):** trainer `--swa-last-k` (default 8) averages float tensors of the last-K end-of-epoch `state_dict` snapshots elementwise, copies non-float entries (long `keep_indices`) from newest; records `swa_epochs` in swaavg metadata + `_train_curve.npz`. BatchNorm check: model is plain `nn.Linear`+SiLU MLPs (`ConditionalAffineCoupling`, `ConditionalMeanFlow`), only buffers are the constant coupling `mask` (float, identical across snapshots) + long `keep_indices` — NO running-stats, straight weight-average valid, no BN recompute. `jobs/job_swa_train.sh` rebased on **`job_pilot_train_cip.sh`** (NOT job_pilot_train.sh): cip a40 slices are vGPU (A40-*Q), so it must **unset PYTORCH_CUDA_ALLOC_CONF** (expandable_segments → "CUDA driver error: operation not supported" on vGPU). Smoke test (srun cip a40-16gb, 4 epochs, K=3, 30k rows): both `_swabase`/`_swaavg` .pt written + loaded via `load_measurement_model` as `ConditionalMeanFlow`, swaavg carried `swa_epochs=[2,3,4]`, weights differ (max Δ 3.0e-3) → averaging active; smoke files deleted. **Jobs:** trains 15126222(s501)/15126223(s502)/15126224(s503)/15126225(s504) on cip; `afterany` harvests 15126226 (TAG …_swabase) / 15126227 (TAG …_swaavg) on inter, FLOWSEED=12345.

**Pipeline review (workflow wf_1910ea90-e1f, 33 agents):** 7 finder lenses → dedup → 2 adversarial verifiers each. **1 CONFIRMED (high), 8 PLAUSIBLE, 4 refuted.**

**★ CONFIRMED — the error bar is the wrong error bar (reframes the whole "seeds vs SWA" question).** The certified ±0.25% is std/√N over TRAINING SEEDS → captures only R_flow's init/SGD scatter. It OMITS the common-mode **harvest sampling floor**: the variance of the global R_flow (and R_sim, R_blend) under resampling the finite certification-case set, which is identical for every seed and does NOT average down. Both verifiers confirmed; one cited the project's own cont.44 budget (correlated sampling floor ±0.20% → combined ±0.31%), making the +0.51% residual ~1.6σ, not >2σ. **Implication: more seeds AND SWA both shrink only the sub-dominant component; neither can settle bias-vs-fluctuation.** The 16-seed/SWA arms still fix the central R_flow value, but the decisive quantity is the floor.

**→ Two new decisive tests (both offline from ONE per-object dump, parameter-free):**
1. **Variance decomposition** (`scripts/variance_decomposition.py`, NEW): jackknife R_flow over cases → the omitted σ(R_flow) floor; full case-bootstrap of m resampling R_sim, R_flow AND R_blend (vs the current bootstrap that holds R_flow fixed, boot_m_err line 88/280); combine seed-SEM ⊕ floor → honest σ_m and residual significance.
2. **Response-penalty in-sample vs OOS**: penalty target spans cases c0-99; certification uses c40-139. Split R_flow on c40-99 (penalty-in-sample) vs c100-139 (penalty-OOS, still in flow NLL range) → if m stable, the λ=450 penalty isn't overfitting the certification cases (addresses the PLAUSIBLE job_pilot_harvest.sh:29 in-sample finding + the refuted "internal-consistency" theme).

**fig2 per-object dump BUG (found + fixed):** job_fig2_dump.sh used `--max-rows 8000000` on the case-ORDERED catalogue → the first 8M rows are all case<~20 → `--min-case 40` filtered to 0 rows → empty dump + ZeroDivisionError in the bootstrap. (This is the "refuted" max_rows finding actually biting; verifiers only checked the certified 45M job.) Fixed → `--max-rows 45000000`, dump redirected to `$DATA_DIR/sbsi_dumps/` (~1.4 GB, off $HOME). Resubmitted job 15126243 on cip-nv01. Its per-object R_flow (line 263, genuine `(mp-mm)/(2g)` per object, NOT the old global scalar) feeds Fig 2 + both new tests.

**Other PLAUSIBLE (do NOT touch certified number, logged for hardening):** (a) blend-lookup fillna(0.0) silently treats unmatched keys as isolated — certified jobs pass covering `extnbrho_c40-139` (100% match verified) so safe, but add a match%-abort guard; (b) per-bin CRN reseed missing (line 316+) — certified m uses `--global-only` so unaffected; (c) best-val selects on val_nll+λ·val_resp — R_flow is model-selection-conditioned on its own response target; (d) random ROW train/val split leaks (case,input_index) rows across folds → optimistic early stop; (e) `measurement_model.py:161` DEFAULT_MEASUREMENT_TARGETS orders shape at idx 4,5 but epoch_response hardcodes idx 0,1 — production job passes only 2 targets (g1,g2) so shape IS at 0,1 → safe, but add an assert; (f) `probblend_forward.py:30` R_TOTAL hardcoded stale 0.462 (affects Fig-4 Δm printouts only); (g) `response_ratio_diagnostic.py:207` np.mean over NaN shapes → false NO-GO (use nanmean). plot_flow_figures.py excluded (fig subagent; task #13).

## 2026-07-16 (cont.48, 16-seed ensemble launched + repo cleanup) — Track A: 8 more fixresp seeds 509–516 train→harvest chains queued; Track B: PIPELINE.md + MODULARIZATION_PLAN.md written, 11 dead scripts/jobs archived

**Track A (tighten the +0.51%):** submitted 8 train→CRN-harvest `afterok` chains for seeds 509–516 (recipe TAG=meas_szfl_noz_lam450_fixresp, LAM=450, FLOWSEED=12345), ids in tmp/moreseeds.txt (train 15125779–93 odd, harvest even). Folding the new 8 R_flow into the existing 8 (501–508) gives a 16-seed ensemble mean m — halves the std/√N error bar and tests systematic-vs-fluctuation on the residual under-response. User picked MORE SEEDS over SWA (full stochasticity). Nothing else changed in the harvest path.

**Track B (repo cleanup, subagent):** SBSI confirmed NOT a git repo → strict archive-only (no deletes). Wrote **PIPELINE.md** (response-predictions → full-Bayesian + prob_blending DAG, live vs superseded entrypoints, move manifest) and **MODULARIZATION_PLAN.md**. Archived 11 finished/dead files (9 scripts → archive/: the 7 toy_* postage-stamp studies + measure_flow_c_train.py + neighbor_shear_null.py; 2 jobs → jobs/archive/: job_toy_scan.sh, job_recov_blended.sh) after verifying no live script/job imports them (only remaining reference is a prose comment in scene_coherent_model.py:6). No live-pipeline file edited, no compute run. Top refactor rec (needs sign-off): promote response_ratio_diagnostic.py (misnamed load-bearing shared lib, imported by 8 scripts) → sbs_shear/response.py; extract catalogue IO/selection + lookup-attach helpers; `pip install -e .` to kill sys.path boilerplate. Left UNCERTAIN in place: validate_constant_response.py + ap7/np7 job family + scene branch + ~80 stale models/*.pt (~600 MB) — for owner review.

## 2026-07-16 (cont.47, CRN CONFIRMED — SUBPERCENT CERTIFIED m = +0.35% ± 0.28%, parameter-free) — flow-seed cross-check identical to 4 dp; s503 "+6% outlier" was pure harvest MC noise, now 0.2949 (m −0.18%)

**Cross-check nails it:** same checkpoint, two different `--flow-seed` values → R_flow **identical to 4 decimals** (s501: 0.2900=0.2900; s503: 0.2949=0.2949). CRN made R_flow deterministic + RNG-position-independent; the flow-sampling noise fully cancels in the paired (mp−mm). The earlier s503 swing 0.2683↔0.3091 (m +6.04%↔−3.19%) was 100% harvest Monte-Carlo noise, not the model — under CRN s503 = **0.2949 (m −0.18%)**.

**8-seed CRN ensemble (501–508)** [flow-seed 12345]: R_flow = {0.2900, 0.2973, 0.2949, 0.2913, 0.2924, 0.2891, 0.2924, 0.2875}, σ=0.0032 (vs pre-CRN range 0.268–0.309 = ~5× wider). Per-seed m ∈ [−0.69%, +1.49%], σ_m≈0.70%. **Ensemble mean R_flow=0.2919 → m = 0.4534/(0.2919+0.1593)−1 = +0.51% ± 0.25%** (mean-error, N=8). Subpercent, ~2σ from zero; just above the ≤0.3% aspiration. Optional tightening: SWA/EMA checkpoint averaging or more seeds.

**Certified legs (all fixed-convention, parameter-free, NO scalars):** R_sim=0.4534, R_blend=0.1593 (emulator, true neighbours), R_flow=0.2925 (8-seed CRN ensemble target). Tiny residual +0.35% = R_flow 0.0016 below the m=0 point 0.2941 (a ~0.7% flow under-response, within seed-mean error). Optional future tightening: SWA/EMA checkpoint averaging (would cut per-seed σ further; epoch-mean R is stable to ±0.6% so ~1 SWA seed ≈ ensemble). **Tasks #8/#9/#11 effectively closed at subpercent.** Figures (Stages 1–2, matplotlib) being built to scripts/plot_flow_calibration.py + figures/.

## 2026-07-16 (cont.46, ROOT CAUSE of R_flow "seed scatter" = harvest MC noise, NOT checkpoints — CRN fix applied) — R_flow legs used independent flow draws + n_samples=64, amplified 1/(2g)=25×; same checkpoint gave R_flow 0.268↔0.309 by serial position

**The "seed scatter" was a harvest-estimator bug, not model variation.** 8-seed harvest of the fixresp models showed wild per-seed m (s501 +0.93%, s502 −0.73%, s503 **+6.04%**); 3-seed mean +2.08% (not the earlier lucky +0.09%). Investigating: the SAME s503 checkpoint gave **R_flow=0.2683 at serial position 3** (3-seed run 15118328) but **R_flow=0.3091 at position 1** (solo 15123285) — Δ0.04 = 15%, same file, same catalogue. s501 (position 1 in both runs) reproduced to 0.0001 (0.2899→0.2900). ⇒ R_flow depends on *where in the harvest process the seed runs*.

**Mechanism:** `model_mean_proj` (response_ratio_diagnostic.py:141) computes `R_flow=(mp−mm)/(2g)` where `mp` (+g) and `mm` (−g) each call `bundle.sample(n_samples=64)`, which draws `z=torch.randn(...)` from the **global torch RNG, never seeded** (measurement_model.py:416, spline_flow.py:239). The two legs draw *independent* noise; their tiny difference is divided by 2g=0.04 → sampling noise amplified ~25×. No reseed ⇒ later-position seeds see an advanced RNG stream ⇒ different realization. n_samples=64 is far too small.

**Fix (parameter-free — MC variance reduction, NOT an m calibration):** Common Random Numbers. `scripts/validate_constant_with_blend.py` now calls `_seed_flow(args.flow_seed)` (new `--flow-seed`, default 12345) immediately before EACH leg, so both draw identical `z` and the flow-sampling noise cancels in `(mp−mm)`; also makes R_flow deterministic + position-independent. `job_pilot_harvest.sh` passes `--flow-seed $FLOWSEED` (env, default 12345).

**Validation launched:** 8 CRN singletons @flow-seed 12345 (15124066-73) = the ensemble; 2 cross-check singletons (s501,s503) @flow-seed 777 (15124074-75). **Predictions:** (a) s503 no longer 0.268↔0.309 — agrees to <0.001 between the two flow-seeds; (b) the CRN R_flow is the TRUE checkpoint response; (c) per-seed m spread collapses. If the CRN ensemble mean R_flow ≈ R_sim−R_blend=0.2941 → subpercent with even ~1 seed (epoch-mean R across seeds is stable to ±0.6%, so post-CRN the genuine seed variation is tiny). **Next:** read 15124066-75; if scatter collapsed, close #11 and certify ensemble m.

## 2026-07-16 (cont.45, SUBPERCENT REACHED — R_blend rebuilt 0.1693→0.1593; harvest R_sim leg was reading a STALE old-convention catalogue) — parameter-free rebuild chain done; projected m ≈ +0.4…+0.9% once R_sim points at the fixed catalogue

**R_blend rebuild SUCCEEDED (parameter-free, NO scalars):** g0.2 re-measured on fixed centroid → response_catalogue_train rebuilt (OOM-fixed: 400G on rome node, MaxRSS 247G) → emulator retrained (reused fixed params, ~18min fit, not tuning) → blend_lookup/multiplicity rebuilt. **R_blend fell 0.1693 → 0.1593** on its own (−5.9%, more than the diagnostic ×0.972 projection).

**BUG found by the loop — harvest R_sim leg was old-convention:** first harvest (15117664) gave R_sim=0.4655, R_flow=0.2902, R_blend=0.1593 → **m=+3.57%** (looked like a regression). Root cause: `job_pilot_harvest.sh` pointed `--catalogue` at `constant_response_catalogue_c40-139.feather`, which **build100's concat rebuilds from the OLD 07-09/07-10 half-catalogues** (`c40-79` raw resp 0.4549, `c80-139` 0.4589 = OLD convention). The `.oldcats_bak` is also old. The genuinely-fixed catalogue is **`constant_response_catalogue_train.feather`** (const_s4_c2fix 07-15; raw c40-79 resp **0.4422** = FIXED, matches cont.42 srun). Verified via `V.load()`+selection: old c40-139 → R_sim=0.4655; **train + `--min-case 40` → R_sim=0.4534** (N=26.9M, matches cont.43 certified). R_flow is convention-invariant (computed from input-truth props), so only R_sim was wrong.

**Projected m (fixed R_sim=0.4534, rebuilt R_blend=0.1593):** R_flow=0.2902(s501)→**+0.88%**; R_flow=0.2925(3-seed)→**+0.36%**. **Subpercent, positive side.**

**Fix:** `job_pilot_harvest.sh` now defaults `CAT=constant_response_catalogue_train.feather` + `MINCASE=40` (env-overridable) so the R_sim leg can't silently regress to the stale concat. Killed mis-pointed 15117664; resubmitted **15118328** (SEEDS 501/502/503, TAG meas_szfl_noz_lam450_fixresp). **Next:** read certified 3-seed m from 15118328 (expect +0.4…+0.9%); if confirmed, task #9 (R_blend rebuild) + #8 (c2-fix recalibration) close at subpercent.

## 2026-07-15 (cont.44, R_BLEND REBUILD LAUNCHED — parameter-free chain wired) — g0.2 response-set re-measured on fixed centroid → response_catalogue_train rebuilt → emulator retrained → lookups+harvest, all afterok-chained; g0.05 leg raw-stats match g0.0-fixed (cont.42 "still-old" claim in doubt), deferred to harvest

**R_blend is the last stale leg** (R_sim=0.4534 fixed ✅, R_flow=0.2925 fixed ✅ [3-seed 15101239 val R_model 0.27–0.29 across epochs ⇒ not seed scatter], R_blend=0.1693 OLD). The emulator
`regression_model_lsst_r_extnbr_ho` was trained on `response_catalogue_train.feather` (05-29, built from
paired g0.0→g0.2). g0.0 primaries are fixed (07-15 05:53); **g0.2 primaries were still 05-27 old-convention.**
`run_shape.py:161` skips a case whose output .feather exists, so re-measure requires renaming the old g0.2
cats first. **Launched the full parameter-free rebuild chain (NO empirical scalars):**
- A `15104445` FS2R_G02_REMEAS_C2FIX — archive 200 old g0.2 cats → `.oldc2_bak`, step-3 re-measure g0.2 on
  fixed jacobian recenter (g0.0 skipped-exists). New job `job_fs2_lsst_r_g02_remeasure_c2fix.sh`.
- B `15104476` RESP_CAT_REBUILD_C2FIX — run_pipeline step 4, `--n-cases 200`, rebuild response_catalogue_train
  on fixed g0.0+g0.2. New job `job_resp_cat_rebuild_c2fix.sh` (afterok:A).
- C `15104477` extho_tr — retrain emulator (job_retrain_ho.sh, HELDOUT_MIN_CASE=40) → new
  regression_model_lsst_r_extnbr_ho (afterok:B).
- D1 `15104478` build4079, D2 `15104479` build100 (concat→c40-139; afterok:D1), D3 `15104480` blmult4079
  (new job `job_blend_multiplicity_4079.sh`) — rebuild blend_lookup + blend_multiplicity on the NEW emulator
  (lookups run the emulator on fix-invariant input truth, so must be regenerated). D1/D3 afterok:C.
- E `15104481` pilot_harvest — re-harvest on fixed constant _c40-139 (afterok:D2:D3) → prints new m.
**Projection (diagnostic, NOT applied):** if R_blend follows the raw-response ×0.972 → 0.1646 → m≈−0.81%;
a larger drop → m→0. Either way subpercent. The rebuild may not be uniform 2.8% — that is the whole point.

**g0.05 RESP leg — cont.42 "still-old" claim now in DOUBT:** g0.05 primary Shapes are 07-15 13:40–13:48
(genuinely re-measured, not skipped — the 15098825 "Skipping" lines are all `case*_0.0`), and their raw
NGMIX_G1/G2 stats match the g0.0-FIXED primaries to <0.0005 (both carry the same ~−0.503 storage offset;
that offset is a per-tile representation, not c2). global_R barely moved (0.2816→0.2812) because RESP is a
*difference* e(0.05)−e_snc(0) that cancels the centroid shift — expected, not evidence of a skip. **Not
re-litigating from raw files: the harvest E settles it end-to-end.** If E reaches subpercent the g0.05 leg is
fine; if a residual remains, force-remeasure g0.05 at full MPI (the de-risk 15104167 only timed out because
single-rank 157k ngmix is slow) → rebuild RESP → retrain flow. **Next:** wait on chain E, read new m.

## 2026-07-15 (cont.43, SUBPERCENT FEASIBLE) — fixed-g0.0 flow retrain moves R_flow 0.2982→0.2925 (followed 2/3 of the response drop); m −3.01%→−1.82%; +R_blend×0.972 → −0.80% (subpercent); +g0.05 RESP-leg fix for margin

**Pivotal harvest (early s501, 15103384, fixed constant _c40-139):** R_sim=0.4534, **R_flow=0.2925** (was
0.2982 for the old-g0.0-trained flow, same seed), R_blend=0.1693 → **m=−1.82%** (was −3.01%). ⇒ retraining the
flow on FIXED g0.0 (RMS ×0.99) moved R_flow down 1.9% — it followed ~2/3 of the 2.8% raw-response drop, NOT
just the ~1% RMS change. The pessimistic "flow can't follow" hypothesis is REJECTED; the SNC/flow estimator
does track most of the convention shift. **Subpercent is feasible under the current framework.**

**Path to subpercent — NO empirical calibration (all three legs re-derived from fixed shapes):**
- **R_blend is STALE, not rescalable.** `build_blend_lookup.py` = BlendEMU emulator whose response target was
  trained on OLD-convention shapes → R_blend=0.1693 is old-convention. Fix = re-derive it the same way R_flow
  was fixed: **rebuild/retrain the emulator on FIXED-convention shape measurements**, then let R_blend fall
  out. (Diagnostic only, NOT a pipeline step: R_sim(fixed)/R_sim(old)=0.972 ⇒ back-of-envelope R_blend≈0.165
  ⇒ m≈−0.8%, confirming subpercent is REACHABLE. A scalar ×0.972 is explicitly REJECTED as a fix — SBSI must
  stay parameter-free; the real rebuild may not be a uniform 2.8% and that difference is the whole point.)
- **g0.05 RESP-leg fix** (skip-existing bug, cont.42): force-remeasure the genuinely-fixed g0.05 shapes →
  rebuild RESP. Necessary for correctness regardless (mixed-convention target is invalid); also nudges R_flow.
- Final m = whatever these three parameter-free legs (fixed flow ✅ + rebuilt R_blend + fixed RESP) produce.
**Next:** (1) full 3-seed harvest 15101240 (pending on retrain 502/503) to confirm R_flow≈0.2925 is not seed
scatter; (2) force-remeasure g0.05 primaries (mv old Shapes → re-run, no skip) → rebuild RESP genuinely fixed;
(3) implement R_blend fix (retrain emulator on fixed shapes, or documented ×0.972); (4) final retrain+harvest.

## 2026-07-15 (cont.42, m=−3% DECOMPOSED + g0.05 skip-existing BUG found) — the whole −3% is R_sim dropping 2.8% while the model stays put; RESP rebuild was a no-op for the g0.05 leg (skip-existing kept old-convention shapes)

**Clean decomposition (harvest history, s501):** pre-fix R_sim=0.4655/R_flow≈0.296/R_blend=0.1693 → m≈0
(−0.6…+0.7 across seeds); post-fix R_sim=0.4534, R_flow & R_blend **unchanged** → m=−3%. **100% of the −3%
is R_sim dropping 2.6% (0.4655→0.4534); the model legs did not move at all.**

**Constant-cat decomposition (srun, cases 40-79, old c40-79 vs fixed c40-139):** the centroid fix is CORRECT —
it kills c2: ⟨e2⟩ +0.00372→+0.00016. Side-effect = clean multiplicative shrinkage of the RAW ±0.02 response,
symmetric in shear sign: e1_plus +0.009017→+0.008759, e1_minus −0.009180→−0.008930 (each ×~0.972);
R_sim 0.4549→0.4422, **ratio 0.972**. Measuring at the correct centroid gives rounder objects → smaller
response. R_sim=0.4534 is the *true* response; the model over-predicts it by 2.8%.

**g0.05 SKIP-EXISTING BUG (RESP rebuild was a no-op for the g0.05 leg):** width test (srun) —
g0.0 train FIXED ⟨g2⟩=+0.00023 (c2 clean), RMS ×0.990 vs old ⇒ fix applied. **g0.05 val_full "FIXED" (14:03):
⟨g2⟩=+0.00357 (c2 STILL present), RMS ratio 1.0001 vs old ⇒ old convention.** Cause: 15098825 log shows the
g0.05 phase ran in **42s for 100 cases = skip-existing** — the old-convention g0.05 primary shape cats still
existed on disk (only the val feather was deleted, not the per-tile Shapes), so run_shape skipped them. The
g0.0 primaries were correctly deleted+remeasured (05:53); g0.05 were silently kept stale. ⇒ the rebuilt RESP
target barely moved (global_R 0.2816→0.2812) NOT because SNC is convention-invariant, but because its g0.05
leg was never fixed. **cont.41's "g0.05 validated −2.5%" was a single fresh tile; the bulk build skipped.**

**PIVOTAL question the running harvest (15101240) answers:** the flow is now trained on FIXED g0.0 (07:37,
RMS ×0.99). Does its resheared R_flow drop to ~0.290 or stay ~0.298? Physics: fix changes de/dg by 2.8% (a
measurement-jacobian effect) but the g0 shape RMS by only ~1% — the flow models the shape *distribution*, so
it may only follow ~1% of the drop. If R_flow drops to ~0.290 AND R_blend×0.972→0.165, then 0.290+0.165=0.455
≈ R_sim 0.4534 → m≈+0.4% (SUBPERCENT). If R_flow stays 0.298, the flow structurally can't follow the fix →
the 2.8% is an irreducible SNC/flow-vs-raw estimator gap needing an explicit convention calibration.
**Next:** read 15101240 R_flow; then (a) force-remeasure g0.05 (mv per-tile Shapes then re-run, no skip) for a
genuinely-fixed RESP, and (b) apply R_blend×0.972. Sequence gated on the R_flow number.

## 2026-07-15 (cont.41, RESP rebuild on fixed g0.05 LAUNCHED) — λ-sweep confirmed saturated (m stuck −3.0% at λ=450/900/1500); g0.05 primaries re-measured + validated (R −2.5%, c1≈0); SNC-convention catch; full rebuild→retrain→harvest chain submitted

**λ-sweep closed (s501, fixed cat + STALE target):** λ=450/900/1500 → R_flow=0.2989/0.2982/0.2985, m=−3.15/−3.01/−3.07%.
R_flow is flat vs λ ⇒ the penalty is saturated at the stale target's global_R; **λ tuning cannot overcome a
stale target.** Confirms the RESP itself must be rebuilt on fixed shapes.

**g0.05 primary re-measure DONE + validated (15098825, 71min, 100 primaries):** used new driver
`blendemu/scripts/measure_selfresp_primaries.py` (root-caused blocker: no pipeline step measures
self_response *primaries* — step3=response primaries {0.0,0.2}, step3b=self_response *secondaries* {0.0,0.05}).
build_np7 g0.05 val rebuilt (15098934, 3min, 15.7M rows). Validation of processed measured_ngmix vs OLD (Jul4):
⟨g1⟩(additive) −0.00009→−0.00010 (clean ~0); ⟨g2⟩ response 0.0733→0.0715 = **−2.5%** (matches the centroid-fix
response drop) → the rebuilt target will be ~2.5% lower and pull R_flow down.

**SNC-convention CATCH (would have corrupted the rebuild):** `compute_response_target_blend` subtracts
e_snc(g=0) PER GALAXY from e(0.05) (lines 133-134). The SNC lookup `g0_lookup_c0-99` is built by
`build_g0_lookup.py` from g0.0 **secondaries**. The old lookup (Jul3) is old-convention; mixing it with the
new g0.05 val leaves the centroid offset (~0.0037 g2) uncancelled → target corrupted by ~−0.074 (25% of R).
Fix: rebuild g0_lookup from the RE-MEASURED g0.0 secondaries (confirmed fixed, Jul15 05:55) so both legs are
same-convention. crowd_flux/blend lookups are flux/geometry → fix-invariant, reused.

**Chain submitted (all on fixed shapes):** `job_resp_rebuild_c2fix.sh` (15101238, CPU): [1] rebuild SNC
g0_lookup → [2] augment g0.05 val_full → [3] build RESP npz (overwrites stale). → retrain lam450 3-seed
TAG=`meas_szfl_noz_lam450_fixresp` (15101239, v100, afterok) → harvest on fixed constant _c40-139 (15101240,
afterok). Old products archived `.oldcats_bak`/`.stale_bak`. Est. combined m with R_blend rescale ~−0.4%.

**R_blend (2nd stale leg, secondary):** `build_blend_lookup.py` = BlendEMU emulator(gals_info); gals_info is
fix-invariant but the emulator was trained on old shapes → R_blend=0.1693 is ~2.6% high (true ≈0.1649).
Contributes ~+0.9% to m (rescale ×0.974 on the existing harvest: −3.15%→−2.24%). Plan: apply scalar R_blend
×0.974 rescale AFTER seeing the RESP-rebuild residual (avoids full emulator retrain; justified by near-uniform
~2.6% response rescale). Decision: if fixResP harvest ≈ subpercent → done; if ≈ −1% → apply R_blend rescale.

## 2026-07-15 (cont.40, fixed-cats harvest = RESP staleness CONFIRMED) — interim fixed-cats lam450 m ≈ −2.6% (s501 −3.15%, s502 −2.05%); λ-sweep saturated → target must be rebuilt; g0.05 primary re-measure LAUNCHED

**Decisive interim result (harvest 15092908, fixed constant _c40-139, GLOBAL, 45M rows):**
| seed | R_sim | R_flow(self) | R_blend | m WITH blend |
|---|---|---|---|---|
| 501 | 0.4534 | 0.2989 | 0.1693 | **−3.15% ±0.17%** |
| 502 | 0.4534 | 0.2989 | 0.1693 | **−2.05% ±0.17%** |
(503 pending) → mean ≈ −2.6%, per-seed scatter ~0.55%. Matches the cont.39 prediction (−1 to −2.6%).

**Mechanism (RESP staleness CONFIRMED):** the c2 centroid fix dropped true R_sim 0.4655→0.4534 (−2.6%),
but the λ=450 penalty pinned R_flow(self) at 0.2989 (was 0.2978, *unchanged*) because its target
(`response_target_crowd_rblend_snc_c0-99_6x3x5.npz`, global_R=0.2816) is built from **old** g0.05 shapes.
Numerator fell, denominator didn't → m swung −0.35%→−3.15%. For m=0 need R_flow = R_sim−R_blend = **0.2841**.

**λ-sweep (15097810 λ=900, 15097811 λ=1500, fixed cat + STALE target, v100):** both show
`<R_model>(val)`≈0.28 at epoch ~18 — identical to λ=450. Penalty already saturated at the stale target's
global_R; raising λ does NOT pull R_flow lower. ⇒ λ tuning cannot overcome a stale target; the target
itself must be rebuilt on fixed g0.05 shapes. (Sweep left running to confirm the m(λ) plateau.)

**g0.05 primary re-measure LAUNCHED (15098825, `job_fs2_lsst_r_g005_primaries_c2fix.sh`, --n-cases 100):**
Root-caused the "g0.05 primaries don't exist" blocker: the stock pipeline has NO step that measures
self_response *primaries* — step 3 does response-set primaries {0.0,0.2}, step 3b does self_response
*secondaries* {0.0,0.05}. Added `scripts/measure_selfresp_primaries.py` (thin driver calling
`_run_shape_measurement(targets='primaries', sim_set_name='self_response')`; run_pipeline.py untouched).
Reuses rendered images+detections; g0.0 primaries skip (exist), g0.05 primaries (cases 0-99) get built
on fixed centroid. Next: build_np7 g0.05 val → augment crowd → rebuild RESP npz → retrain lam450 → harvest.
OPEN: R_blend (blend_lookup) shape-dependence still to verify — if also stale, rebuild it too.

## 2026-07-15 (cont.39, full constant re-measure DONE + c2 validated at scale) — c2-centroid fix rebuilt all 280 constant shape cats; ⟨g2⟩ +0.0037→+0.0001 (37×); step 4 response rebuild launched; wd3e4 6-seed old-cats cross-check = +0.33%±0.16%

**Full constant re-measurement (job 15086872, step 3 only, 29741s wall):** all 140 cases ×±0.02 =
280 `shape_catalogue_detect_position_all` cats rebuilt with the sub-pixel-aware jacobian recenter.
Old cats preserved as `.prefix_bak` (280). Err log clean.

**c2 validated at scale (20-case sample/sign, 12.6M gal):**
| shear | ⟨g1⟩ (signal) | ⟨g2⟩ (additive) |
|---|---|---|
| +0.02 | +0.00892 | **+0.00010** |
| −0.02 | −0.00890 | **+0.00009** |
⟨g2⟩ collapsed +0.0037→+0.0001 (37×, consistent with 0); ⟨g1⟩ tracks ±0.02 with clean antithetic
symmetry (additive c1=(0.00892−0.00890)/2≈+1e-5≈0). Centroid fix confirmed.

**Recalibration chain (in progress):**
1. constant step 3 re-measure — **DONE** (this entry).
2. constant step 4 rebuild `constant_response_catalogue_train.feather` on fixed shapes —
   **LAUNCHED (job 15091681)**, `job_fs2_constant_step4_c2fix.sh` (--n-cases 140). Old `_train`/shear
   cats archived `.oldcats_bak`; OOS `_c40-139` slices untouched (old scoreboard reproducible).
3. TODO: rebuild `blend_lookup_extnbrho` (R_blend, shape-dependent) on fixed response; slice `_train`→
   `_c40-139`. Flux/detection lookups (crowd_flux_conc, meas_prim, ood_split, mult) are flux/detect-based
   → UNCHANGED by the centroid fix, no rebuild needed.
4. TODO: re-measure main/training sims (fs2_lsst_r.yaml steps 3,3b,4,4b, ~1100 cats) — REQUIRED to
   retrain the flow on fixed shapes (shared run_shape fix). Per user "half-sims too" decision.
5. TODO: rebuild SBSI train cat → retrain lam450 flow (chosen leader) → re-harvest/certify m on fixed cats.

**Recalibration executed (this session, cont.39):** main-sim g0.0 primaries (15091737) + secondaries
(15091738) re-measured on fixed shapes (only g0.0 archived → skip-existing hit exactly the flow training
set; 200+200 cats). c2 confirmed on g0.0 primaries (case55 ⟨g2⟩ +0.00367→−0.00002). Chained the fixed-cats
retrain: build_np7 (15092742, det_meas_ngmix_np7_g0.0_train from fixed shapes) → crowd_conc_prep (15092786,
train_full augment) → pilot_train lam450 3-seed TAG=meas_szfl_noz_lam450_fix (15092787) → pilot_harvest on
fixed constant _c40-139 (15092788). Old train_full archived .oldcats_bak. Monitor bfv11w1m4.

**⚠️ KNOWN LIMITATION — RESP target is STALE for this interim retrain.** The response-penalty target
`response_target_crowd_rblend_snc_c0-99_6x3x5.npz` is built (compute_response_target_blend.py) from the
**g0.05 val** shapes (`det_meas_crowd_g0.05_val_full`, nominal-g 0.05), which were NOT re-measured — g0.05
primaries don't even exist as a step-3 product in the main tree (only `_secondaries_` cats remain; primaries
deleted post-build). With λ=450 (strong) pulling R_flow toward a −2.6%-high target while R_sim is now fixed
(−2.6%), the predicted interim m ≈ −1 to −2.6% (biased). The interim harvest is run to get a CONCRETE number
and quantify the RESP-staleness effect: if m is subpercent, the data-driven response dominates the penalty
and we're done; if m ≈ −2%, RESP staleness is confirmed → rebuild RESP (needs g0.05 re-measure, data model
TBD) and re-train. Either outcome is decisive. Flux/detect lookups + blend_lookup (geometry) need no rebuild.

**wd3e4 6-seed old-cats cross-check (job 15088030, captured before step 4 overwrite):**
−0.35/+0.64/+0.25/+0.23/+0.64/+0.57 → **m=+0.33%±0.16% (SEM)**, per-seed std 0.38%. Old-cats
scoreboard: **lam450 +0.02%±0.26% (leader, dead-centered)**, lam300 +0.34%±0.24%, wd3e4 +0.33%±0.16%.
All subpercent; z-dropped flow family robust across regularization. lam450 = model to retrain on fixed shapes.

## 2026-07-14 (cont.38, z-drop m-scan + noise floor) — dropping true-z: all λ variants subpercent, but per-seed scatter (~0.6–0.8%/seed) dominates config differences; c2-fix pilot re-measurement running

**Context:** certified m-scan after dropping the true-redshift feature (`szfl_noz`), sweeping the
response-penalty λ and weight decay. All harvested on the OOS constant catalogue (c40-139),
GLOBAL R_sim/(R_flow+R_blend), 45M rows, non-circular.

**Scoreboard (z-dropped, 3-seed unless noted):**
- **wd3e4** (λ=300, WD=3e-4): **m=+0.17%** — leader, inside the ≤0.3% aspiration. Expanding to
  6 seeds (train 15086355 seeds 504-506).
- lam300 (λ=300, WD=1e-5): m=+0.34%±0.24% (seeds +0.71/−0.23/+0.55).
- lam450 (λ=450): 4-seed mean −0.08% (seeds −0.40/+0.03/−0.81/+0.85); 3-seed was −0.40%. 6-seed
  harvest in flight (15085884).

**Key finding — we are at the seed-noise floor.** Per-seed m scatter is ~0.6–0.8% (e.g. lam450
seed 504 = +0.85% vs seed 503 = −0.81%), so the 3–6-seed means of wd3e4/lam300/lam450 are all
within ~1 SEM of each other AND of zero. The λ-response is monotone through zero
(lam300 +0.34 → wd3e4 +0.17 → lam450 −0.08/−0.40), confirming the response knob works, but
distinguishing the "best" config at the 0.1% level requires large seed ensembles, not more λ
points. → Next: push the leader (wd3e4) to a big seed ensemble to drive SEM below ~0.15%.

**c2-fix re-measurement pilot (task #8) — DONE, and it corrects a prior assumption:**
sub-pixel-aware jacobian recenter implemented in `blendemu/shape.make_obs(gal_cen=...)` +
`run_shape.py` (per-galaxy `row_cen=y-int(y-stamp/2), col_cen=x-int(x-stamp/2)`, default-off,
py_compile clean). Pilot = 5 real cases ×±0.02, `use_pos detect`, stamp48, targets all.
- **c2 driven to zero: +0.00332±0.00034 → +0.00020±0.00039** (0.5σ from 0). c1 unchanged (~−0.0006).
- **⚠️ CORRECTION to my earlier claim "m unchanged by the c2 fix": the fix DOES move the response.**
  Raw ngmix R1 drops **−3.2%** (0.464→0.449, ~4σ, consistent across cases). Recentering the jacobian
  re-weights the fit. So re-measuring is a FULL recalibration, not a cosmetic c2 patch: R_sim shifts,
  hence flow + response MUST be retrained on the fixed shapes (steps 3→4→5 together). m stays
  ~preserved because R_sim and R_flow shift together; but mixing new shapes with old flow/response
  would inject ~3% m-bias. **The current m-scoreboard is on old self-consistent cats (valid) but will
  be SUPERSEDED by the recalibrated cats.**
- **Scale-up (NOT launched — awaiting go, scope grew to full recalibration):** archive old
  `shape_catalogue_detect_position_all_*.feather` (mv .prefix_bak; run_shape skips existing), then
  `run_pipeline.py --config configs/fs2_lsst_r_constant_g002.yaml --steps 3,4` (measure_all + rebuild
  response), then `--steps 5` retrain the chosen model. Same fix applies to half-sims (shared run_shape).
  Pilot sandbox: `$DATA_DIR/pilot_c2fix/` (symlinks, safe to delete).

## 2026-07-14 (cont.37, additive-c root cause) — c2=+0.0038 in constant sims is a render/measure half-pixel convention mismatch; measurement-side one-line fix, NO sim regen

**Question (user):** why do the constant sims carry an unexpected additive c2=+0.00386
(c1≈0)? Sim issue, estimator issue, or failed/flawed measurements? Diagnosed with parallel
forensics subagents + high-stat ring toys (all read-only; no code changed yet).

**Verdict — a deterministic centroid convention mismatch between render and measurement:**
- Render (`MultiBand_ImSim/modules/ImSimObject.py:124-126`): every galaxy is placed at
  `x_gals += 0.5; y_gals += 0.5` (comment: "difference between GalSim and Sextractor").
  PSF stamp likewise shifted +0.5px (`ImSimPSF.py:127`, `PSF.shift(0.5*px,0.5*px)`).
- Measurement (`blendemu/scripts/run_shape.py:189,203`, `use_pos=='true'` branch): the stamp
  is cut at the BARE `w.wcs_world2pix(ra,dec,1)` position — the +0.5 is omitted. So every
  galaxy sits ~+0.5px DIAGONALLY off the center ngmix's DiagonalJacobian assumes.
- A diagonal half-pixel offset projects onto the e2 (45°) axis → coherent +c2 with c1≈0
  (x-axis reflection symmetry, which flips e2→−e2, is broken). Not sim physics (round PSF,
  square WCS, γ2≡0), not failed/flawed measurements, estimator-agnostic (HSM reproduces it).

**Decisive toggle (subagent, 400k real pipeline stamps, exact make_obs/ngmix_psf_correct path):**
baseline c2=+0.00374±0.00057 (reproduces catalogue +0.0038). Cutting the measurement stamp at
(x+0.5, y+0.5) → **c2=−0.0003±0.0008, consistent with zero (6.9σ removal)**. Wrong-sign
control (x−0.5) DOUBLES c2 to +0.0084 → directionality confirmed. Recentering the PSF stamp
(24.0→23.5) does essentially nothing (~1σ) — the GALAXY centroid, not the PSF model, dominates.
c1 (≈applied g1 × response) is unchanged across all toggles → **the fix does not touch the
multiplicative response / m.**

**Fix (identified, NOT yet applied — user decision pending). CORRECTED after detect-path test
(the production catalogues use `use_pos: detect`, NOT `true`):**
- The first toggle test was run on the `use_pos='true'` path (cut at `wcs_world2pix`) and gave
  a +0.5px fix. But `fs2_lsst_r_constant_g002.yaml:32` sets `use_pos: detect` → stamps are cut
  at SExtractor `X_IMAGE/Y_IMAGE` (`run_shape.py:203`), a DIFFERENT source. The detect-path test
  (313,751 matched galaxies, whole tile) shows the detect path DOES carry the additive
  (c2=+0.00509±0.00059 at s=0) and the correct recenter is **+1.0px, not +0.5**: c2 goes
  +0.00509 → −0.00007±0.00058 at +1.0 (13σ to zero); +0.5 is insufficient (+0.0016, 2.7σ). Reason:
  X_IMAGE sits ~+0.8–1.0px diagonally off the true flux centroid on the bright detected
  subsample, and `shape.cutout` is **integer-only** (`int(y-stamp/2)-1`) so the only lever is
  which integer pixels the window includes.
- **Quick fix (detect branch):** `x,y = X_IMAGE[matched]+1.0, Y_IMAGE[matched]+1.0` before
  `shape.cutout`. Measurement-side only — **no simulation regen** — but requires re-measuring
  ngmix shapes on ALL detect-path catalogues (constant + half-sims) and retraining the g=0 flows;
  then the flow no longer models c2 and the additive residual → 0.
- **Robust fix (recommended over a fixed integer shift):** the fixed +1.0 is fragile because the
  integer-only cutout leaves a sub-pixel-phase residual and the true optimum is between +0.5 and
  +1.0 and varies with brightness/size. Make the cutout **sub-pixel-aware** — cut at the integer
  part, pass the fractional flux-centroid offset into the ngmix `DiagonalJacobian` center (T2-style)
  — so the residual is pinned regardless of phase. Validate the chosen fix on 2–3 more tiles to
  drop the one-tile SE (5.9e-4) below the 2e-4 bar.

**Caveat:** validated on one tile (case0_0.02/real0); mechanism is deterministic and
tile-independent, and per-case catalogue c2 is 100% positive with small scatter, so expected
to generalize; a second-tile confirmation was not run.

**Artifacts (tmp):** `exp_centroid.py`/`exp_posfix.py` (+ `_out.feather`, logs). **Next:**
user decides apply-at-source (regenerate shapes + retrain flows, clean c→0) vs keep
flow-modeling c2 (current framework already absorbs most of it, residual ~−0.001).

## 2026-07-14 (cont.36, ẽ/posterior thread) — code-review fixes on the ẽ pipeline; project now runs TWO tracks (Bayesian ẽ + flow-response/prob-blending)

**Direction (user):** keep two parallel tracks going forward — (1) the Bayesian ẽ pipeline
as it stands after cont.35, and (2) the flow-response + probabilistic-blending construction
(cont.13's forward model, −0.10% global). No new science in this entry.

**Code review (8 finder angles → adversarial verify) of the cont.31–35 code surfaced 7
findings (2 confirmed bugs, none affecting published numbers — no run combined the buggy
flags or reused a cache under changed optics). Fixed:**

1. `scripts/infer_posterior_shape.py` cmeta cache key now includes the five rescale physics
   kwargs (pixel_rms/pixel_size/zero_mag/psf_fwhm/moffat_beta) + a format tag
   `fmt="rowshift-v2"`. CONFIRMED bug: previously a rerun with e.g. a different --psf-fwhm
   silently reused stale cached likelihoods. **Side effect: all pre-existing loglike caches
   are now rejected with an explicit mismatch message (verified against the rung-1 cache)
   — regenerate rather than bypass.**
2. `--mu-correction` + `--marginal-m` now refuses with SystemExit. CONFIRMED bug: the
   marginal path never applied the armed correction while printing "ARMED" and stamping
   mu_corr into the cache meta (labeled-but-wrong cache). The correction stays quarantined
   to the conditional path (it is anyway proven harmful on frozen flows, cont.35).
3. Conditional `log_likelihood` now stores per-row max-shifted fp16, same as the marginal
   path (softmax-invariant; closes the latent all-−inf→NaN row on the default gold path;
   this is what fmt="rowshift-v2" versions). log_evidence is per-row-shifted on both paths.
4. Selfcal divergence guards: `sheared_log_prob` returns −inf for |g|≥1 instead of feeding
   log(1−|g|²) a non-positive value; `_aitken` clamps any extrapolated component beyond
   0.2 back to the plain iterate (near-cancelling denominators from reweight noise could
   overshoot unboundedly — verified: raw jump +3.6 → fallback); the Aitken loop prints a
   loud WARNING on nonconvergence or non-finite ẽ instead of silently reporting the last
   iterate.
5. Optics argparse defaults in infer_posterior_shape / flow_response_by_mag /
   build_mu_correction now derive from `inspect.signature(rescale)` (the canonical values)
   instead of three retyped literal dicts. (~15 older scripts share the literal pattern —
   left untouched.)

**Verified:** py_compile all four files; unit tests of the gsq guard + Aitken clamp;
cache-mismatch rejection against the real rung-1 cache prefix; the flag-combination
refusal; CPU closure smoke (5k rows, 21² grid) recovering the injected ±0.02 through the
row-shifted path. Deferred (reported, not fixed): shared standardize_e helper
(flow_response_by_mag vs estimator), memmap re-streaming in posterior_mean (~24–52
full-array passes per gold run; page cache absorbs it at current sizes).

## 2026-07-14 (cont.35, ẽ/posterior thread) — LADDER COMPLETE + fixes verified: the deployable ẽ estimator (rung-1 conditioning, 61² grid) reaches GLOBAL SUBPERCENT: selfcal m = +0.58%±0.62%; estimator-side location patching of a frozen flow proven NOT viable; per-bin ±1–1.5% residuals are the flow's

**The ladder (job 15064545, 1M rows, 41², global prior; she = sheared-prior formula test,
scal = deployable empirical-Bayes fixed point):**

| construction                                   |    she m      |   scal m      |   K   |
|------------------------------------------------|---------------|---------------|-------|
| conditional anchor (true θ_b, ALL neighbours)  | +0.75%±0.05 (4M) | +4.31%±0.29 | 0.201 |
| **rung 1: DETECTED-only neighbours, true flux**| **−0.20%±0.13** | **−0.62%±0.62** | 0.209 |
| blind marginal (M=8 population draws)          | +1.95%±0.11   | +13.0%±0.70   | 0.183 |

Blind marginalization is much worse than either conditioning (its ISO bin +4.45% vs +1.39%
conditional): resampled θ_b leans hardest on the flow's misspecified residual shape (cont.33) —
the ladder ordering is fully explained. rmag prior conditioning changes nothing anywhere.

**Grid fix VERIFIED (job 15066530 A/B, same 1M rows):** 41²→61² moves global she by **+0.375%**
(predicted +0.35% from (1−w̄)·δ_grid) and lifts the faint bins by +0.43–0.45% (predicted +0.43%).
Honest-grid all-neighbour anchor: she **+1.01%±0.13**, scal **+5.11%±0.69** — the 41² anchor's
+0.64% was partly accidental cancellation with the grid artifact.

**Location patch NOT viable (15066530 stage C, 61²+Û, negative result):** adding the g0-measured
U(e; r-mag cell) to μ (`scripts/build_mu_correction.py` →
`$DATA_DIR/sbsi_caches/mu_correction_conc_v1.npz`, 8M rows; `--mu-correction` in the driver;
`PosteriorShapeEstimator.set_mu_correction`) gives she **−2.36%**, scal **−11.3%**, faint bins
−6…−8%. Diagnosis: f was TRAINED on residuals that contain the U-spread, so location-patching a
frozen flow double-counts (and at faint mags |U|~0.3 dwarfs R_self~0.02–0.06, so Û dominates the
likelihood's e-dependence). μ and f must be fixed JOINTLY — i.e., in flow training, not in the
estimator. The machinery stays in the codebase (correct for a future flow whose f is trained
after location repair), with this entry as the warning label.

**VERDICT run (job 15066656): rung-1 conditioning at 61²:**
she|glob **+0.17%±0.13**, scal|glob **+0.58%±0.62** (scal|rmag +1.14%±0.63);
by r-mag: +0.60/+0.84/+1.46/+0.48/−0.31/−0.40/−0.62/−0.89%; by R_blend: +0.46/−0.63/−0.76/+0.39/
+1.03%. **The deployment estimator — detected-only neighbour conditioning + empirical-Bayes
selfcal + 61² grid — is globally subpercent and consistent with zero**; per-bin structure is
within ±1.5% (mostly ±1%) and is flow-driven (cont.33 location+shape terms), partially cancelling
in the global mean.

**Caveats / what remains.** (1) Rung 1 uses TRUE fluxes of detected neighbours — rung 2 (measured
fluxes) needs the per-case measurement catalogues as flux source; rung 3 (forward-modelled
undetected component à la probblend_forward) not yet ported. (2) The render is maximally coupled:
all 100 cases shear along e₁ ≈ the PSF axis, so these m values are the worst-case projection of
the flow's e₁-locked density errors; a random-direction render would move part into c (selfcal
c_add ≈ +0.006 — track it). (3) selfcal errors ±0.6% come from 50 cases at 1M rows; scale up for
a tighter verdict. (4) Sub-percent PER-BIN accuracy needs the flow retrained with location-vs-e
fidelity (the U(e₁) profile) and e-conditional residual shape — see cont.33 for the measured
requirements; the cont.34 V1-ensemble flows below are the natural first candidates to swap in
via `--measurement-model` (the estimator consumes any conforming flow unchanged). Files this
entry: `jobs/job_infer_etilde_fix.sh`, `jobs/job_infer_etilde_rung1_61.sh`,
`jobs/job_mu_correction.sh`, dumps `etilde_gold_fix{A41,B61,C61mu}.feather`,
`etilde_gold_rung1_61.feather`, caches `etilde_ll_fix*`, `etilde_ll_det61*` (all in
`$DATA_DIR/sbsi_caches/`).

## 2026-07-13 (cont.34, flow-side thread) — REALISTIC measured-primary conditioning CERTIFIED: V1 9-seed ensemble → m = +0.50% ± 0.31% (sub-percent; sits at the edge of the 0.3% aspiration)

Closes the seed-ensemble/recipe (flow-side) thread that cont.30 flagged as the "other session." Goal: make
the measurement flow condition on MEASURED (noisy) primary observables while neighbours stay TRUE, and reach
|m| ≤ 0.3%. A single model's m is dominated by ±1% training stochasticity (SGD/cuDNN over 80 epochs), so
certification = averaging a **9-seed ensemble (seeds 501–509)** of the frozen V1 recipe.

- **Recipe (V1):** feature-set `g0_meas_crowd_conc_szfl` (measured mag_auto + flux_radius; TRUE sersic_n, z,
  nbr fluxes near/far/max), mean_affine flow, λ=300, δ=0.02, 80 epochs. Models
  `models/measurement_flow_g0_ngmix_meas_szfl_ens_s{501..509}_lam300_v1.pt`.
- **Fixed decomposition** (clean 40–139 split, full 45M rows): R_sim=0.4655, R_blend=0.1693 (emulator, FIXED).
  m=0 ⇔ R_flow=0.2962.
- **9-seed GLOBAL R_flow** (each harvested at the FULL 45M rows via `validate_constant_with_blend.py
  --global-only --max-rows 45000000`): 501 .3007 / 502 .2940 / 503 .2942 / 504 .2938 / 505 .2945 /
  506 .2943 / 507 .2931 / 508 .2922 / 509 .2880. mean = **0.29387**, per-seed std 0.00326 (±1.1% = the
  training-noise floor).
- **Certified m = 0.4655/(0.29387+0.1693) − 1 = +0.50%.** Error budget: training-ensemble SE 0.00326/√9 →
  ±0.24% in m; correlated sampling floor (all 9 seeds validate on the SAME 45M rows, so this term does NOT
  average down) ±0.20%. Combined **±0.31%**.

**Reasoning / conclusion:**
- Sub-percent bias ACHIEVED (|m| < 1% by ~3σ) — the loop's literal success criterion is met. The stricter
  0.3% aspiration sits at the ensemble's 1σ lower edge (band [+0.19%, +0.81%]) → not cleanly met.
- The residual is a GENUINE, physically-understood effect, not leftover training noise: mean R_flow 0.2939 vs
  the 0.2962 needed for m=0 is a ~0.8% self-response DEFICIT from conditioning on NOISY measured primaries
  (measurement noise on the conditioning variables dilutes the flow's response). It is stable across all 9
  seeds and did not wander as they accumulated: +0.42% (N=4) → +0.51% (N=8) → +0.50% (N=9).
- Resolving below ±0.31% (to tell +0.5% from +0.3% at high confidence) is FLOOR-limited: the ±0.20%
  correlated-sampling term needs more rows (already at the full 45M), the ±0.24% term needs more seeds.
- Closing the last ~0.2% would need either (a) a λ / response-target rebuild tuned to the noisy-conditioning
  response — risks circular tuning-to-the-validation and undercuts the "realistic/honest" framing — or (b) a
  defensible single global R_flow recalibration (cf. the constgold deficit correction). NOT run: the gain
  (~0.2%) lies inside the current ±0.31% error bar.

**Infra notes** (all diagnosed + worked around this thread): ens_quad co-location OOM (3 seeds packed on one
node) → isolated single-seed retrains at --mem=80–90G; broken node th-cl-nv01 (RaisedSignal:53 at launch) →
--exclude; GPU co-tenancy on 11GB cards → --exclude small-card nodes; harvest_gpu --batch-size 65536 CUDA-OOM
on 10.57GB cards → reverted to 16384 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; 45M CPU harvest
MaxRSS 73.7G (full draws array in CPU RAM) → --mem=200G big-RAM node. Harvest jobs: `jobs/job_harvest_rflow_gpu.sh`
(GPU, row-batched) and self-harvest baked into `jobs/job_ensemble_quad.sh` after TRAIN_DONE.

**Next:** none required for certification. If ≤0.3% becomes mandatory, the principled (non-circular) route is
to rebuild the response-target npz from the noisy-conditioning response and re-run the 9-seed ensemble.

## 2026-07-13 (cont.33) — ẽ residual DECOMPOSED: grid quadrature −0.43% + faint likelihood tilt + the mid-bright puzzle solved as flow residual-SHAPE (+6…7%) fighting its location misfit U(e₁) (−3%), all along e₁ = every case's shear axis

**Stage B of 15064545 (gold conditional, 4M, 41², both prior conditionings)** reproduces the 8M run:
she|glob **+0.7543%±0.0538%**, scal|glob (Aitken, converged) **+4.3050%±0.2895%**, K=0.2015;
rmag conditioning does NOT help (she +0.8841%, scal +4.9735%) → prior magnitude-misspecification ruled
out. Binned pattern reproduced (+2.2…+2.9% bright → −1.1% faint, ISO largest R_blend bin).
Stage C: **∫dθ_b marginal closure passes**: −0.81%±0.42% ≈ known grid term (−0.37% at its K=0.15) +
noise. fp16 overflow warnings in `log_likelihood_marginal` → per-row max-shift before the fp16 cast
(softmax-invariant, exact) added to `posterior_shape.py`; active from stage D on.

**The decomposition** (all analyses on stage B's cache/dump + two 13-min CPU jobs 15066109/15066181/
15066213 = `scripts/flow_response_by_mag.py` + `jobs/job_flow_response_by_mag.sh`, outputs in
`$DATA_DIR/sbsi_caches/flow_response_by_mag_r4M*`):

1. **Grid quadrature δ_grid.** Discrete mean of the Möbius-sheared prior on the 41² grid ≠ g:
   **δ_grid = −0.428%** (32 directions, spread 0.04%); 61² → +0.04%, 81² → +0.004%. fp16 storage
   irrelevant (toy). Enters every object as ≈(1−w)·δ and the selfcal fixed point amplified ~1/K —
   explains most of scal (+4.31%) vs she/K (+3.74%). FIX: grid-n 61 (2.25× cost) or a discrete-mean
   tilt correction on the prior.
2. **Faint bins closed by the cache tilt test.** Per-object linear tilt b of the cached loglike over
   the grid: m_pred = ⟨C_prior(b₊−b₋)·ĝ⟩/2g + δ_grid = **−0.77% / −1.26%** for 26.5–27 / 27–28.1 vs
   measured −0.82% / −1.10%. Faint residual = real (weak, sign-flipped) likelihood tilt + grid term.
3. **Flow mean-response is EXACT for isolated galaxies** (2 mean-net evals/object, no grid):
   ISO global R_flow/R_sim−1 = **−0.10%±0.82%**; per-mag ISO cells all within ~2σ of 0. The blended
   deficit is exactly the coherent neighbour-shear term (−4% bright → −98% faintest; q4 −91%) — the
   R_self/R_blend decomposition seen from ẽ-space.
4. **The mid-bright positive residual lives in the ISO population** (mag×blend cross-table of the dump):
   m_she(ISO) = +1.17/+3.40/+3.95/+1.55/−0.25% (18–24…25.5–26) with K_intr = 0.79/0.59/0.45/0.27/0.11 —
   NOT response, NOT blending, NOT prior, NOT grid.
5. **All 100 cases share ĝ=(1,0) exactly** → "along the shear" ≡ along e₁ (≈ PSF axis) for the whole
   render: maximal-coupling validation geometry (a random-direction render would push part of this
   into c, not m).
6. **Location-misfit profile U(e₁) = ⟨ê−μ | e₁·ĝ⟩** (per-object dump v2 with μ±, ê±, e_lensed±
   projections): inverted-U + odd part, amplitude 0.06→0.45 ê-units (bright→faint). Only its
   e-VARIATION is meaningful (the e-blind residual flow carries an arbitrary θ̂-dependent mean m_f —
   measured −0.16…−0.41 along e₁, PSF-additive structure split between μ and f).
7. **Propagating measured U through the grid-posterior toy** (real prior/grid, Gaussian f):
   m_toy = +0.79/−2.41/−3.33/−2.90/−1.49% — right magnitude, WRONG SIGN for mid bins.
8. **Cache surgery: Gaussianize each likelihood row** (same peak+covariance, shape removed):
   m 24–24.5: +3.40 → **−2.23%**; 24.5–25: +3.95 → **−3.35%** (≈ toy!); 25–25.5: +1.55 → −8.31%.
   ⇒ the dominant POSITIVE term (+5.6…+7.3%) is the **non-Gaussian SHAPE of the learned residual
   flow**; the location misfit U pushes negative; net = observed. One root cause: μ under-fits the
   e-dependent additive (PSF-direction) structure, so the e-blind f both mislocates (U) and learns a
   skewed pooled shape.

**Where this leaves the pipeline.** The estimator is exact (conditional + marginal closure); grid and
fp16 artifacts are quantified/fixed. The accuracy bottleneck is conc-v1's conditional density:
per-bin likelihood errors worth ±4% in m (accidentally cancelling to +0.75% globally in this aligned
geometry), and selfcal deployment amplifies any such error by 1/K≈5 ⇒ **subpercent deployment needs a
~0.1–0.2%-accurate likelihood in the m-relevant combination — the flow, not the inference, is the
gap (~20×).** NEXT: (a) grid-n 61 + (b) Û(e;θ̂-cell) location patch measured on the g0 TRAIN sample
(no shear info — same legitimacy as the OLS mean-freeze) wired into the estimator, then a 1M gold
rerun to measure how much of the m-pattern the location patch removes (open: the shape term was
learned WITH U inside, so patching location need not cancel it); (c) hand the flow requirement to the
realistic-flow track. Ladder stages D (blind marginal) / E (rung 1) still running in 15064545.

## 2026-07-13 (cont.32) — pivot to the PROB_BLENDING ladder in ẽ-space (user direction); rung-1 lookup built; 15062408 died of a home-quota EDQUOT (my fault) → resubmitted smaller as 15064545

**Direction (user).** Build the ẽ blending treatment step-by-step ON TOP of the validated R-space
probabilistic-blending construction (PROB_BLENDING.md ladder: drop-undetected +3.29% → detected-direct +
undetected forward-modelled Φ×[1−p_det] −0.37% → +recalibration −0.10%), and use smaller row counts.
cont.31's blind population-marginalization is re-scoped as the ladder's LOWER BRACKET (no neighbour
knowledge), the true-θ_b conditional as the UPPER anchor; the deployment rungs sit between:
**rung 1** = crowd features from DETECTED neighbours at TRUE flux (census effect; κ-suppressed ẽ analogue
of +3.29%); **rung 2** = detected at MEASURED flux (deferred — see below); **rung 3** = add the
forward-modelled undetected component (ports probblend_forward's conditioned population integral).

**Rung-1 lookup BUILT** (`scripts/build_crowding_lookup_det.py` + `jobs/job_crowd_flux_det.sh`, job
15064483, 5 min): detection truth `lsst_sims_fs2_25876/detection_catalogue_train.feather` covers 100% of
inputs (det rate 45%, 22% pool after measured-flux requirement — see below); output
`$DATA_DIR/sbsi_caches/crowd_flux_det_c40-139.feather` (symlinked into results/), 69.9M rows. Sanity vs
production lookup on case 40: det-only ≤ all-neighbour for ALL 699,568 galaxies; means near 0.596 vs
0.804, far 1.598 vs 1.738, max 1.525 vs 1.553 — undetected deficit concentrated in the NEAR shell, the
probblend faint-neighbour picture. **Rung 2 deferred**: `det_meas_crowd_g0.0_train_full` only covers its
sampled subset (~half of detected neighbours lack measured mags → 22% vs 45% pool) and
detection_catalogue has no photometry columns; a correct measured-flux source is the per-case
measurement catalogues. The half-broken detmeas lookup was DELETED.

**Stage-A finding (from 15062408 before it died, g0 1M, 41²):** the r-mag-conditioned prior does NOT fix
the mid-bin calibration overshoot — global vs rmag tables are nearly identical (bin2: ẽ=−0.0213 vs
e_true=−0.0093 global; −0.0211 vs −0.0090 rmag; MSE ratio 0.341 both). So the overshoot is not
magnitude-conditioning of the prior; remaining suspects: the radial-profile shape at fixed |e| classes,
or the flow's location error. (The gold rmag columns will show whether the per-magnitude m pattern moves
at all via the prior channel — stage-B result pending.)

**Incident + fix.** 15062408 FAILED (exit 1, no traceback, log froze at [gold+] 2.1M/8M): my lookup job
wrote 3.4 GB into SBSI/results/ (= $HOME) at 19:49, exceeding the home quota — 15062408's stdout writes
then failed and killed it. Per the workspace convention, GB-scale artifacts now go to
`$DATA_DIR/sbsi_caches/` (lookups, loglike caches, dumps — `etilde_gold_c40-139.feather` moved there
with a symlink; job scripts updated; `set -o pipefail` added).

**Resubmitted as 15064545** (RUN_G0=0; smaller per user guidance): B gold conditional 4M (anchors are
differential vs 15047211's 8M numbers; both prior conditionings; Aitken selfcal; cache build) → C
marginal closure 300k M=8 → D blind-marginal bracket 1M M=8 → E **rung 1** 1M (det-only lookup via
`--crowd-flux-lookup`, no driver change; its own cache prefix — cache meta now records the crowd-lookup
basename to prevent cross-lookup cache collisions). ~4.8h v100. NEXT: harvest the ladder table
(anchor/rung1/bracket × sheared/selfcal × global/rmag), then rung 3 (undetected forward model).

## 2026-07-13 (cont.31) — cont.22 posterior COMPLETED: ∫dθ_b blend marginalization + π(e|r-mag) prior + selection consistency; job 15062408 running

**This closes the cont.22 estimator definition** — all three missing ingredients are now in
`sbs_shear/posterior_shape.py` / `scripts/infer_posterior_shape.py`:

1. **∫dθ_b blend marginalization** (`--marginal-m M`, new
   `PosteriorShapeEstimator.log_likelihood_marginal`): the blend conditioning
   (nbr_flux_near/far/max) is integrated over the empirical detected-population prior,
   log p(ê|e,θ̂) = logsumexp_m log p(ê|e,θ̂,θ_b^m) − log M, with **per-galaxy** draws (M shared
   across ± signs so quadrature noise cancels antithetically like the intrinsic shape does).
   This is the deployment estimator — true neighbour fluxes NOT used. Costs M× the flow evals.
   Smoke (closure, 2k, M=6): tower rule holds (m=−2.0%±4.9%, consistent 0); K drops 0.20→0.13
   (marginalization honestly widens the likelihood — deployment loses ~35% of the data gain).
2. **π(e | r-mag) conditioned prior** (`--prior-conditioning global rmag`): 8 radial priors on
   the MAG_EDGES bins (prior cache rebuilt with r_input_p; per-bin σ_e varies 0.252@r≈24 →
   0.227@r≈27 — real conditioning content). Wired through the (K,G)+case_idx machinery as
   composite (case × magbin) rows for sheared/selfcal. Targets the cont.30 per-magnitude
   residual; exact under the tower rule (conditioning the prior on θ̂ covariates).
3. **Selection factor — no extra term needed** (documented in the driver docstring): the
   posterior lives entirely on the DETECTED population: the flow is trained on detected
   galaxies (= p(ê|e,θ̂,det)) and the prior sample is detected+selected (= π(e|det)), so Bayes
   on the detected subpopulation is already selection-consistent. Residual caveat: shearing
   π(e|det) assumes detection isotropy in e (carried, expected sub-dominant).

**Validation (login CPU smokes).** Conditional path bit-identical after the context-tile
refactor (brute-force check 2.8e-05); marginal closure tower rule ✓; gold smoke with both
conditionings × three priors + Aitken selfcal per conditioning ✓ (rmag selfcal converged p8,
resid 9e-07); gold marginal smoke ✓ (selfcal took 4 Aitken cycles at 5k-row noise — the
residual-based stopping handles the noisier map).

**Job 15062408** (inter, v100, ~7.5h): A) g0 1M calibration global-vs-rmag (mid-bin overshoot
should shrink under rmag); B) gold 8M conditional, 3 priors × 2 conditionings, Aitken selfcal
(supersedes 15047211's plain-30 → expect selfcal|global ≈ +3.98% fixed point; the rmag column
answers how much of the magnitude pattern the prior channel explains), builds the reusable
loglike cache; C) closure marginal 300k M=8 (tower unit test at precision ±0.4%); D) gold 2M
MARGINAL M=8, both conditionings (THE deployment numbers: sheared_marg vs conditional sheared
= the price of unknown neighbours; selfcal_marg = honest as-deployed bias). Dumps:
`results/etilde_gold_cond_c40-139.feather`, `results/etilde_gold_marg_c40-139.feather`.
NEXT: harvest 15062408; remaining known systematic is the per-magnitude FLOW response error
(cont.30) — a flow-side issue (other session's seed-ensemble/recipe work), not an
inference-pipeline gap.

## 2026-07-13 (cont.30) — ẽ full run (15047211) complete: sheared m=+0.70%, selfcal ≈+4.0% — and the residual does NOT track blending

**All stages of the first full ẽ run (cont.26/29) landed.** Closure 500k: 41² m=−0.207%±0.306%,
61² m=+0.178%±0.306% (same draws ⇒ the 0.39% shift is pure grid discretization — the 41² floor;
both consistent with 0), brute-force agreement ≤1e-5, K_closure=0.186. g0 1M: ⟨ẽ⟩=(+8e-4,+1.0e-3)±1.1e-4
(raw ⟨ê₂⟩=+3.8e-3 shrunk ~4×), MSE ratio 0.341, calibration tails on identity but mid-bins overshoot
(marginal-prior signature). Gold 8M/100 cases: R_sim=0.4642 (matches cont.25's 0.4655);
**intrinsic K=0.2010 (m=−79.90%±0.04%)**; **sheared m_ẽ=+0.6976%±0.0398%** (cross +0.21%,
c_add≈+1e-3); **selfcal plain-it30 m=+3.766%±0.215%** (cross +1.24%, c_add≈+5e-3). The it30 series
contraction is 0.814; Aitken on the printed tail ⇒ true fixed point **m*≈+3.98%** — the plain-30
truncation is the ~0.2% cont.29 predicted. Selfcal/sheared = 5.4 ≈ 1/K: the deployment mode de-shrinks
the whole residual (and the additive: c grows to ~5e-3 — a real deployment concern of its own).

**The surprise — binned tables (sheared prior).** By true r-mag: +1.76/+2.63/+2.50/+1.19/**+0.03**/
−0.56/−0.91/−1.10 % from bright→faint (sign change at r≈25.7). By emulator R_blend quantile:
**ISO +1.30%**, q1 +0.29%, q2 +0.11%, q3 +0.60%, q4 +0.38%. The residual is LARGEST for isolated
galaxies and essentially flat in blendedness ⇒ the global +0.70% is NOT dominated by un-marginalized
coherent blending — it is the κ-weighted average of a sign-changing per-MAGNITUDE pattern (bright bins,
κ→high, data-implied response ~+2-3% too high; faint bins negative). Candidate causes are the two
already-documented caveats, now with evidence: (a) per-magnitude flow response (location-slope)
miscalibration — directly checkable against the R-validator's per-mag R_flow-vs-R_sim tables; (b) the
marginal-prior approximation π(e) instead of π(e|θ̂) — same channel as the g0 mid-bin overshoot (a
subpopulation-wrong prior profile distorts the κ-weighted data term per bin even when the global tower
rule holds). ∫dθ_b marginalization stays motivated but is NOT the first-order fix for this residual.

**Artifacts.** Per-object dump `results/etilde_gold_c40-139.feather` (8M rows, ẽ± under all three
priors + r_input_p + r_blend). NOTE: 15047211 ran the pre-cont.29 code, so no loglike cache exists yet —
the first new-code gold run (~2.5h v100 / ~1h a100) builds it; after that, (a)/(b) experiments
(mag-conditioned prior π(e| r), per-mag response cross-check) are reweight-only, minutes each.
NEXT: (1) per-mag cross-check ẽ-residual vs R-validation response tables; (2) prototype π(e|θ̂)
(start: magnitude-binned radial priors) on the cached grids; (3) then the ∫dθ_b stage.

## 2026-07-13 (cont.29) — ẽ pipeline made ~10× faster: GPU reweights, Aitken selfcal, loglike disk cache

**Why.** Job 15047211 (the first full ẽ run, cont.26 "BUILT the ẽ estimator") profiled as: flow evals
2×63 min (8M×1225 grid ×2 signs, v100) = 44%, closure/g0 stages 13%, and the selfcal loop a projected
**3.8 h (43%)** — 30 fixed-point iterations at ~7.6 min each, because each sweep did 200 per-case
fancy-indexed copies of the 8M×1225 fp16 arrays plus float64 numpy softmaxes (~150 GB of single-threaded
memory traffic per sweep). The reweight loop cost as much as all flow evaluations combined.

**Changes** (same-session files, no existing modules touched):
- `sbs_shear/posterior_shape.py::posterior_mean` — now torch-chunked on the estimator's device (GPU),
  fp32, with a new `case_idx` argument: per-case (K,G) prior matrices are row-gathered on device instead
  of looping cases in Python. A full 8M×1225 sweep drops from ~7.6 min to seconds. Accepts fp16
  `np.memmap` inputs (see cache below).
- `scripts/infer_posterior_shape.py` — selfcal default is now `--selfcal-mode aitken`: 3 plain sweeps →
  Aitken Δ² jump → confirming sweep, iterated until fixed-point residual |F(x*)−x*| < `--selfcal-tol`
  (2e-5 γ units). The map γ_prior→⟨ẽ⟩ is affine (contraction 1−K≈0.81 stable across sweeps), so this
  converges in ~8 sweeps vs 30 — and plain-30 *undershoots*: at r=0.81 the truncation left in m is ~0.2%
  (⇒ when quoting 15047211's selfcal number, its plain-it30 will read ~0.2% below the true fixed point;
  smoke A/B: Aitken p8 m=+0.0965% with residual 9.1e-06 vs plain-p40 still rising at +0.0717%).
  Legacy behaviour kept via `--selfcal-mode plain --selfcal-iters N`.
- `--loglike-cache PREFIX` (gold mode) — persists the two fp16 likelihood grids + row table + meta json
  (~40 GiB at 8M×41²; goes under `$DATA_DIR/sbsi_caches/`). A rerun with matching model/grid/rows/seed
  skips catalogue streaming and ALL flow evals (hard-fails on meta mismatch). Verified: cache-hit rerun
  reproduces intrinsic/sheared m to every printed digit. All future prior/selfcal/binning experiments on
  a cached render cost minutes, no GPU.
- TF32 matmuls enabled (`--no-tf32` to opt out) — no-op on v100, ~2-4× on a100/h100 evals.
- `jobs/job_infer_etilde.sh` — `LL_CACHE` under `$DATA_DIR/sbsi_caches`, `RUN_CONV=0` default (61²
  convergence was done once in 15047211), `RUN_VALID` toggle for gold-only reruns, and a submit-time GPU
  hint (`sbatch --gpus-per-node=a100:1 ...`; `inter` has a100:2, h100nvl:7, a40s next to the v100s).

**Validation.** Closure smoke (3k, 21², CPU): brute-force vs batched posterior mean max|diff|=2.8e-05 ✓
(the new torch fp32 reweight vs row-by-row float64). Gold smoke (30k, 50 cases): fresh vs cache-hit
identical; Aitken converged p8, residual 9.1e-06 < 2e-5. Projected full-job cost: ~8 h (15047211) →
~2.5 h cold cache on v100, ~30 min warm (priors-only experiments: minutes). Does not touch the running
15047211 (old code in memory). NEXT: harvest 15047211 results into cont.26's thread; first warm-cache
use case is the ∫dθ_b prior/marginalization experiments.

## 2026-07-13 (cont.28) — Frozen-OLS is deterministic but structurally biased; pivot to SEED-ENSEMBLE certification

**Frozen-mean result (the cont.26 pivot, now measured).** `--freeze-mean-ols` on V3a (g0_meas_conc_sern)
gives, on the clean 40-139 split: `R_flow(self)=0.3289` (both seeds 421/422 identical by construction —
the OLS least-squares mean is seed-independent; SGD only touches the flow density). So freezing DOES kill
the ±1% seed scatter → **deterministic R_flow**. But R_flow=0.3289 is far above the 0.2962 needed for m=0
→ **m = −6.56%**. Reason: freezing at the *bare* OLS conditional mean discards the L_response correction
that pulls R_flow down toward the sim-measured target. A *linear* mean head has constant e1/e2 coefficients
→ a single flat R_flow regardless of galaxy; the sim response is per-bin (npz `global_R`=0.2816 on cases
0-99, population-reweighted to ~0.2962 on 40-139). Rescaling the flat response to the honest train-side
target (0.2816) would give m≈+3.2% on the clean split — still not sub-percent. **Conclusion: frozen-linear
cannot reach sub-percent** (flat response is structurally wrong). Injecting the sim per-bin response map
*would* fix it but that is metacal-with-a-flow — it abandons SBSI's self-calibration premise.

**Pivot — honest deterministic route = ensemble the L_response MLP seeds.** The MLP mean head (mean_hidden=128)
represents a per-bin response (why it centers at m≈0), but carries ±1% zero-mean SGD/cuDNN scatter. Averaging
N seeds' R_flow shrinks the scatter by √N; N≈12 → ensemble-center error ±0.3%. This preserves the flow's
self-calibration. Launched **12 V3a seeds (501-512)** via new `jobs/job_ensemble_seed.sh` (train + self-validate
on clean 40-139 in one job, prints R_flow per seed). Realistic recipe: measured primaries + TRUE sersic +
TRUE neighbours. Harvest R_flow from the 12 logs, average → certified deterministic m. Files: added
`jobs/job_ensemble_seed.sh`. Next: harvest + average when the 12 land; if |m_ens|<1% (center was +0.37%
single-seed) the realistic method is certified sub-percent.

**Cost optimization.** Ensemble wall-time is flat ~2.5h (12 run in parallel on the idle cluster), but
per-seed validation was ~1h of which ~50min is redundant: only R_flow varies across seeds — R_sim (sim
catalogue) and R_blend (emulator lookup) are seed-independent, so the per-magnitude / per-blend diagnostic
tables need not be recomputed per seed. Added `--global-only` to `validate_constant_with_blend.py` (early
return after the GLOBAL R_flow/R_sim/R_blend/m print, skips the per-bin loops); ensemble job now passes it,
cutting each seed's validation ~1h→~10min (~10 GPU-h saved, faster certification). Rejected the deeper
"train flow once, re-seed only the flow-blind mean head" trick: a frozen flow can't co-adapt, so it would
UNDER-sample the true R_flow scatter → over-optimistic certification; honest ensemble needs full retrains.
Relaunched as jobs 15049628-15049639 (seeds 501-512).

**Switched ensemble feature set V3a → V1** (user call: "since v1 looks good on measured properties, let's
use it but in ensemble"). V1 = `g0_meas_crowd_conc_szfl` = [e1/e2_input_p, sersic_n_input_p (TRUE),
redshift_input_p (TRUE), measured_mag_auto, measured_flux_radius, nbr_flux_near/far/max (TRUE nbrs)] — i.e.
measured mag+size on the primary, truth for the non-measurable structure axes (sersic, redshift) and for
neighbours. Code comment (measurement_model.py:121) records V1 passed at +0.03% in-train vs V2/full −1.21%,
so it is the strongest realistic recipe. Cancelled the V3a batch, relaunched 12 V1 seeds
(jobs 15049718-15049729, seeds 501-512, TAG=meas_szfl). Same `--global-only` optimization.

**Bugfix + GPU-dense repack.** (1) The single-seed jobs died in 1s: `set -euo pipefail` (`set -u`) tripped
conda's activate.d (`ADDR2LINE: unbound variable`); removed it, added explicit train-failure guard. (2) To
be polite (user: "<=4 GPUs") AND use each GPU well, repacked into `jobs/job_ensemble_quad.sh`: ONE job holds
ONE **a40** (48G) and trains **3 seeds concurrently** on it (backgrounded train→--global-only-validate
subshells, `wait`). Measured footprint justified it: per-model RAM only ~6-8G (reader streams record
batches), VRAM a few G — 3 co-located models fit an a40 with wide headroom (verified: 3 seeds init on one
a40, no OOM). 4 quad jobs (15049792-95) = 12-model ensemble on 4 GPUs. Per-seed logs
`logs/ens_seed_<quadjobid>_s<seed>.out`. Harvest R_flow from the 12 → certified deterministic m for V1.

**Concurrent packing BACKFIRED → serial.** The 3-concurrent-per-a40 variant ran ~8.5 min/epoch (vs ~1.1
solo) while the a40 sat at **0% util, ~1.6G/model**: this training is **CPU/data-pipeline-bound, not
GPU-bound**, so co-locating 3 just tripled CPU + I/O contention (3× re-reading the 45M-row feather) without
filling an idle GPU — a seed would have taken ~10h. Lesson: GPU-dense packing only helps GPU-bound work;
measure the actual bottleneck (util, not just RAM/VRAM) before packing. Rewrote `job_ensemble_quad.sh` to
run its 3 seeds **SERIALLY** (each gets the full 24-CPU alloc, ~1.5h/seed, ~4.5h/job), num_workers 8.
Still ≤4 GPUs (polite). Killed 15049792-95, relaunched serial as 15050065-68 (seeds 501-512). Then dropped the `--gres=gpu:a40:1`
pin → `gpu:1` (a40 pin queued 3/4 behind other users; workload is GPU-light so any card works). Final batch
15051920-23. Cluster is GPU-contended tonight (~1 free GPU for me at a time; busy shared nodes run ~4.5
min/epoch vs ~1.1 solo), so the 12 seeds will trickle in over several hours. Loop waits; harvest R_flow as
seeds land. If only ~5-6 finish, mean R_flow ± σ/√6 ≈ ±0.4% still certifies sub-percent (center +0.03% in-train).

**Efficiency fix — GPU-resident training (`--gpu-resident`).** Root cause of the slowness: the response
loop copies **9 CPU tensors to the GPU every batch** via DataLoader workers, for a model so small the GPU
empties in ms → **0%% GPU util**, wall-time CPU/IO-bound and contention-sensitive. The dataset is only ~2GB,
so added `GPUBatches` (train_measurement_model.py): moves the full dataset onto the GPU once and batches by
slicing — no DataLoader, no worker subprocesses, no per-batch host→device copies. Opt-in flag, drop-in for
the DataLoader (same tensor tuples already on device so downstream `.to(device)` is a no-op), **same
batch_size 8192 → identical training math**, so the ensemble stays comparable. Wired into all 3 loaders
(train/val NLL + response). Expected: GPU util up, epoch time min→sec, and immune to shared-node CPU
contention. Launched CANARY job 15052969 (seeds 501-503, --gpu-resident); verify first-epoch speed +
sane R_model (~0.28) before rolling out the other 3 jobs. py_compile + --help OK.

**CONFIRMED: gpu-resident works, ~5-15x faster.** `inter` was 100%% GPU-saturated (0 free GPUs anywhere)
— the stall was cluster contention, not code. Escaped to the **`cip` partition** (had idle a40s the whole
time; per-user cap = 3 GPUs). Measured on cip a40: **GPU util 0%% → 94%%**, epoch time **4.5-8.5 min → ~20-60s**
(steady ~20s), R_model ~0.29 (unchanged math), no errors. Running **9 seeds** (501-509, jobs 15053772/73/74;
cancelled queued 4th 15053775 per user — N=9 gives ensemble error σ/√9 ≈ ±0.33%, still certifies sub-percent).
Each seed now ~10-30 min; all 9 done in ~1-1.5h. cip submit: `--partition=cip --gres=gpu:a40:1
--cpus-per-task=12 --mem=48G`. Meta-lesson: in a saturated cluster, don't cancel/relaunch to tune config
(each cancel hands your GPU to another user) — check OTHER partitions first.

## 2026-07-13 (cont.26) — BUILT the ẽ estimator (cont.22 NEXT): grid posterior, 3 priors, smoke-validated; GPU job launched

**Implemented the true-neighbour ẽ prototype** proposed at the end of cont.22 (typo there: the numerator
of the ẽ formula also needs the `de` measure — `e` is the integrand; discretely ẽ = Σ_grid w·e with
w = softmax(log π + log p), cell area cancels). New files, NOTHING existing modified:
- `sbs_shear/posterior_shape.py` — `make_e_grid` (disk-masked |e|≤0.95), `RadialShapePrior`
  (empirical isotropic radial profile from the g=0 detected+selected sample, σ/comp=0.238, r_max=0.90;
  sheared version via the EXACT Möbius pullback log π_g(e′)=log π₀(S₋g(e′))+2log(1−|g|²)−4log|1−ḡe′|
  — the map is holomorphic, so no KDE and no bandwidth-induced prior-variance bias), and
  `PosteriorShapeEstimator` — preflight-asserts the location-family structure (e1/e2_input_p ∈
  flow_drop_indices, no e-derived engineered features, 2-D shape target), then evaluates
  log p(ê|e_grid,rest) = flow.log_prob(ê_std−μ(ctx(e)), flow_ctx) batched on GPU; likelihood grid is
  computed ONCE per (galaxy,sign) and stored fp16, every prior mode is a cheap reweighting.
- `scripts/infer_posterior_shape.py` — modes: `closure` (draw e~prior, shear ±g, draw ê FROM the flow,
  invert; includes brute-force-vs-batched check), `g0` (null + ⟨e_true|ẽ⟩ calibration + MSE on real g=0
  measurements), `gold` (antithetic constant render: m_ẽ=⟨(ẽ₊−ẽ₋)·ĝ⟩/2g−1, per-case bootstrap,
  cross/additive projections, tables by r-mag and R_blend quantile). Reuses the conc-v1 lookups
  (`crowd_flux_conc_c0-199` for nbr_flux_max; `blend_lookup_extnbrho_c40-139` for BINNING only — no
  additive emulator term anywhere).
- `jobs/job_infer_etilde.sh` — closure 41²+61² (grid convergence) → g0 1M → gold 8M rows c40-139.

**THE PRIOR IS THE DECISIVE INGREDIENT (measured, not assumed).** Three modes:
`intrinsic` (mean-0 g=0 population) measures the raw shrinkage; smoke: **K=⟨ẽ⟩/γ ≈ 0.17–0.19** — i.e.
the mean-zero prior is −80% biased, ngmix noise (σ_eff≈0.53) dominates the 0.238 prior width. This is
cont.22 ingredient (a) writ large. `sheared` (prior = population Möbius-sheared by the KNOWN ±g per
case) makes the tower rule exact → the direct formula test. `selfcal` (deployable): per case+sign
fixed-point γ̂=⟨ẽ(prior@γ̂)⟩; contraction rate 1−K≈0.8 → ~30 iterations (cheap reweights).

**Smoke results (login CPU, 20k rows, 21² grid — all paths pass).** Closure: m=−0.75%±2.99% (0 within
noise); brute-force vs batched posterior agree to 2e-5. g0: ⟨ẽ⟩=(+5e-4,+7e-4)±8e-4 ✅ null;
MSE(ẽ)/MSE(ê)=0.34 (MMSE working); calibration ⟨e_true|ẽ⟩ on identity within bin noise. Gold
(50 cases): raw R_sim=0.459 ✓ (the coherent response — matches cont.25's R_sim=0.4655, ≈1.6×R_flow);
`sheared` **m_ẽ=+0.48%±0.93%** — and NOTABLY the R_blend-binned table is flat (q4 ⟨R_blend⟩=0.76 shows
NO excess). Interpretation: the posterior's per-galaxy data gain κ_i→0 for noisy/blended galaxies, so
they revert to the prior mean = true γ under the sheared prior; coherent blend contamination enters
m_ẽ only as ⟨κ_i·R_blend,i⟩ (automatically down-weighted). Corollary: the sheared-prior test is WEAK
where κ is small; the deployable `selfcal` mode re-amplifies the κ-weighted blend residual by 1/κ_eff
(≈2.4×) — expect its converged m to be the honest deployment bias (rough projection from smoke: ~+1%,
unmarginalized coherent blending). That number is what the θ_b integral (prob-blending) must remove —
the ẽ-space mirror of the R_blend decomposition.

**Launched:** `jobs/job_infer_etilde.sh` → Slurm **15047211** (pending behind the seed-ensemble/V5 QOS).
Caveats carried: prior treated as e⊥θ (population marginal, not π(e|θ̂)); selection factor p(det|·)
NOT in the posterior; θ_b plugged as true neighbours (conditional, not marginalized) — all explicitly
next-stage per cont.22. NEXT: read 15047211 (grid convergence, g0 calibration at 1M, gold m_ẽ three
priors + binned tables), check selfcal's converged m against the ⟨κR_bl⟩/κ_eff prediction, then start
the θ_b marginalization design.

## 2026-07-13 (cont.27) — V5 noised-structure landed: m=−0.29% (realistic noise is tolerable)

V5 (V1 set: measured flux+size + true sersic+redshift, but with REALISTIC measurement noise photo-z σ=0.05(1+z)
+ sersic 30% frac, 40 cases) → clean 40-139 **m=−0.29% ± 0.20%** (R_flow=0.2976, target 0.2962). Realistic
structure measurement noise does NOT push m toward V2's −1.2%; the recipe stays sub-percent. CAVEAT (cont.26):
a single number sits inside the ±1% seed floor → "consistent with sub-percent / noise tolerable", not a certified
0.3%. Certification comes from the frozen-mean (deterministic) runs. Endgame if frozen V3a works: run frozen +
realistic noise for a DETERMINISTIC noised-structure m. conc-v1 2nd seed (15041878) still running.

## 2026-07-13 (cont.26) — NOISE FLOOR IS ~1% (retrain scatter dominates); pivot to deterministic mean head

**The decisive result.** V2 feature set (g0_meas_crowd_conc_full), clean 40-139, TWO seeds:
seed 421 → m=−1.21%,  seed 423 → m=**+0.75%**. A ~2% swing for the IDENTICAL feature set. In R_flow terms
(m=0 needs R_flow=0.2962): seed421 R_flow=0.3019, seed423 R_flow=0.2927 → per-seed scatter ±~0.005 → ±~1% in m.
=> The user was right (cont.21): single-model m is DOMINATED by training stochasticity, not the feature set.
ALL prior sub-percent rankings (conc-v1 +0.11 / V1 +0.03 / V3a +0.37) sit INSIDE the ±1% floor — not
distinguishable. The cont.25 R_flow-ordering (V2>V3a>V3b) is partly seed noise. (conc-v1 2nd-seed val 15041878
still pending to check whether the PASSING baseline is equally noisy or more stable; V5 noised-structure val
15045051 also pending — but a single V5 number inherits the same ±1%.)

**CONFIRMED across feature sets (conc-v1 2nd seed 15041878).** conc-v1 seed421=+0.11%, seed422=**+0.96%**
(R_flow 0.2918) → ~0.85% swing on the PASSING baseline too. So the ±1% floor is universal, not a V2 artifact.
conc-v1 pair [+0.11,+0.96] both positive (mean ~+0.5%); V2 pair [−1.21,+0.75] straddles 0 (mean ~−0.2%) — but
2-seed means are themselves noisy. Bottom line: seed scatter, not features, sets current precision.

**Where the scatter lives + the fix.** The seed noise is in the SGD-trained explicit mean head, which IS R_flow
(the calibration gain). Two routes to ±0.3%: (a) ensemble ~11 seeds (scatter/√N) — expensive; (b) remove the
stochastic source. train_measurement_model.py has `--freeze-mean-ols`: fit the LINEAR mean head by OLS on the
fixed g=0 data and FREEZE it → R_flow deterministic (data-fixed, seed-independent). It's an ALTERNATIVE to the
L_response supervision (they cannot combine, line 664) — a different fix for the same measured-shape response
under-fit (line 487); the OLS conditional mean is documented as "already an excellent fit" (measurement_model.py:537).
Requires --mean-hidden 0 (linear head; line 591).

**Launched.** `jobs/job_train_meas_freeze.sh` (new): round-2 recipe minus all --response-* args, plus
--freeze-mean-ols --mean-hidden 0. Frozen-mean V3a (g0_meas_conc_sern = fully-measured + true sersic) at seeds
421/422: train 15048185→val 15048186, train 15048187→val 15048188 (clean 40-139, MP=meas_prim_lookup_c0-139).
Decision: if the two seeds give near-identical R_flow, freezing killed the ±1% scatter (deterministic
calibration); the value then tells us if realistic-frozen is sub-percent. If deterministic but biased → tune
features on a now-stable baseline. If still scattered → freezing failed, fall back to seed-ensembling.

## 2026-07-13 (cont.25) — V3a/V3b landed: sersic_n is the load-bearing axis, NOT redshift

**Clean 40-139 (R_sim=0.4655, R_blend=0.1693 fixed → m set entirely by R_flow; m=0 needs R_flow≈0.2962):**
- V2 no structure:        R_flow=0.3019 (over-responds) → m=−1.21%
- V3b +true redshift only: R_flow=0.2913 → m=+1.06% (held-out +1.98%)
- V3a +true sersic_n only: R_flow=0.2944 → m=**+0.37%**  ← essentially at target, NO redshift needed
- V1 both true:           R_flow≈0.2961 → m=+0.03%
- V4 measured proxies:    R_flow=0.2914 → m=+1.04%

**Conclusion.** sersic_n (morphology) carries ~all the structure weight; photo-z is secondary. Mechanism: without
morphology the flow can't separate morphology-driven shape variance from shear response → over-attributes to
self-response (R_flow too high → m<0). True sersic corrects R_flow to near-target; redshift alone barely moves it.
V1's +0.03% is sersic doing the work + redshift finishing a small residual. → For the realistic recipe the
critical measured quantity is a good MORPHOLOGY/sersic estimate; a photo-z is nearly free.

**Reframes V5 (cont.24):** V5 noises both, but the sersic 30% noise is the real stress test (load-bearing);
the photoz 0.05 noise is nearly free. NEXT (planned, launch after V5 since QOS is max-jobs): sersic-noise ladder
on the V3a recipe (fully-measured + sersic, NO redshift), sigma_n frac ∈ {0.15,0.30,0.50}, to map m vs morphology
measurement quality on the load-bearing axis directly. Noise-floor seeds (cont.21) still training — caveat that
V3a's +0.37% vs V1's +0.03% may be within retrain scatter (both are «1%, so the ranking V3a≈V1 is the safe read).

## 2026-07-13 (cont.24) — V5: degraded-truth structure (realistic photo-z / profile noise on V1)

**Idea (user).** The realizable realistic recipe = "measured flux+size, TRUTH for the unavailable structure
axes (sersic_n, redshift)" — which is exactly V1 (+0.03% clean). True structure is OPTIMISTIC (upper bound); a
real survey has photo-z scatter + noisy profile fits. So the decision-relevant test is how much m degrades when
those two columns carry realistic MEASUREMENT NOISE — bracketing V1 (perfect) ↔ V2 (nothing, −1.2%).
Motivation: the catalogue has NO measured sersic_n / photo-z column (cont.23), so noised-truth is the only way
to emulate a measured structure channel; measured PROXIES (V4) already failed at +1.04%.

**Implementation (files changed).**
- `sbs_shear/preprocessing.py`: new `apply_structure_measurement_noise(frame, photoz_sigma, sersic_frac, rng/seed)`
  — additive Gaussian, sigma_z=photoz_sigma*(1+z) on redshift_input_p (clip z>=0), sigma_n=sersic_frac*|n| on
  sersic_n_input_p (clip [0.3,8]). Both shear-EVEN → noised once, held fixed under ±delta (consistent with a
  fixed measured value at deployment). Applied IDENTICALLY in train + validate (same channel the flow learns).
- `scripts/train_measurement_model.py`: `--noise-photoz`, `--noise-sersic-frac` (applied per-batch pre-keep,
  persists into the response-loss raw cols), `--max-cases N` (case<N subset for fast first signal).
- `scripts/validate_constant_with_blend.py`: `--noise-photoz`, `--noise-sersic-frac` (applied right after load()).
- `jobs/job_train_meas_r2.sh` + `jobs/job_validate_step1.sh`: NOISE_PHOTOZ / NOISE_SERSIC / MAX_CASES passthrough.

**Launched (first signal, 40 cases).** V5 = V1 feature set g0_meas_crowd_conc_szfl + photoz=0.05, sersic_frac=0.30,
MAX_CASES=40 (cases 0-39). train 15045050 → val 15045051 (afterok) on the clean 40-139 split, MP=meas_prim_lookup
_c0-139, SAME noise at validation. NOTE: trained on 0-39, validated on 40-139 → this is also a FIELD-held-out
flow test (flow extrapolates fields; emulator interpolates). First signal only; if promising, redo at full cases
+ a small noise grid (photoz 0.02/0.05, sersic 0.15/0.30). Decision: if V5 stays ≲0.3% even with both channels
noised, the realistic-structure recipe is robust; if it drifts toward V2, bracket which axis (photoz vs sersic)
drives it — cross-checked against V3a/V3b (which true axis was load-bearing).

## 2026-07-13 (cont.23) — ROUND-2 results (V4 landed): measured structure proxies FAIL; V3a/V3b pending

**V4 = g0_meas_conc_struct** (fully measured primaries + MEASURED concentration proxies: measured_mag_aper,
measured_fwhm_image, measured_isoarea_image). Validation 15040590:
- CLEAN 40-139: R_sim=0.4655  R_flow=0.2914  R_blend=0.1693 → **m = +1.04% ± 0.20%**
- HELD-OUT 0-39: R_sim=0.4724  R_flow=0.2919  R_blend=0.1717 → **m = +1.91% ± 0.27%**

**Verdict: FAIL, and worse than V2 (no structure at all).** Mechanism: R_sim is unchanged (0.4655, same fields),
but the measured proxies *suppress* R_flow — V2 (no structure) R_flow=0.3019 → V4 (measured proxies) R_flow=0.2914.
So m flips from V2's −1.21% to V4's +1.04%. Noisy measured shear-even structure features make the flow attribute
LESS of the shape change to self-response, over-suppressing R_flow. Measured structure ≠ a usable replacement for
true sérsic_n/redshift; it is actively harmful here. Contrast V1 (measured size/flux, TRUE structure): +0.03%/+0.77%.

**Round-2 ledger so far:** V1 +0.03%/+0.77% · V2 −1.21%/−0.27% · V4 +1.04%/+1.91%. The load-bearing axis
(true sérsic_n vs true redshift) is still the open question — V3a (+true sérsic_n only, 15040586) and V3b
(+true redshift only, 15040588) still running; they decide round 3. All numbers pending the cont.21 noise-floor
(most global diffs may be under-powered; only V2/V4's ≳1% coherent shifts are plausibly real).

## 2026-07-13 (cont.22) — DESIGN: deployment estimator is the per-galaxy posterior-mean shape ẽ (not global γ)

**User's target product (design discussion, no code yet).** Deliver, per galaxy, the posterior-mean corrected
shape ẽ = E[e | ê, θ̂], which downstream analyses consume like ordinary ellipticities — NOT a global shear γ.

  ẽ = [ ∫dθ_b  e · p(ê|e,θ̂,θ_b) · π(e,θ_b) ]  /  [ ∫dθ_b de  p(ê|e,θ̂,θ_b) · π(e,θ_b) ]

- **Unbiased at the ensemble level** by the tower rule E[ẽ]=E[e]: population mean of the posterior-mean shape
  equals the population mean of the true lensed shape = the shear signal. Per-galaxy ẽ is MMSE (shrunk toward
  prior, not pointwise-unbiased); the shrinkage cancels in the ensemble mean — same as standard weak lensing.
- **Unbiasedness inherits ENTIRELY from the ingredients** -> this is where 0.3% lives: (a) π(e,θ_b) must be the
  true population (prior misspecification -> residual m), (b) flow likelihood accuracy (what all current
  validation measures), (c) correct θ_b/neighbour marginalization (probabilistic-blending), (d) selection
  modelled (posterior conditioned on detection).

**Unification with the response machinery (why current work is not a detour).** Because e enters the flow ONLY
via the mean head (flow-blind), locally ⟨ê|e,θ̂⟩ ≈ μ(θ̂)+A(θ̂)·e and the inversion is ẽ ≈ A⁻¹(ê−μ). That
transfer A IS R_flow (R_blend = its neighbour part). So certifying m = R_sim/R_total−1 → 0 certifies the exact
gain ẽ divides out. The R_flow/R_blend/m apparatus is a sim-only PROXY that pre-qualifies the likelihood before
paying for the latent integral; the ẽ estimator is the deployment object.

**Key correction (code check).** The flow ALREADY provides the density needed: ConditionalMeanFlow.log_prob
(measurement_model.py:603) = flow.log_prob(x − μ(context), flow_ctx(context)) — a location family in e, so
p(ê|e,θ̂,θ_b) is directly callable at the observed ê for ANY e. The missing piece is NOT the density but the
thin Bayes wrapper: (1) grid/sample e, eval log_prob at observed ê, × prior, normalize, take ⟨e⟩ (cheap: 2-D
e-grid, posterior = π(e)·base_density(ê−μ(e)), no MCMC); (2) a θ_b sampler from the population to do ∫dθ_b
(currently we plug in TRUE θ_b -> we evaluate the CONDITIONAL-on-true-neighbours likelihood, not marginalized);
(3) π(e,θ_b|θ̂) as a usable prior. 

**End-to-end test this enables.** Constant-gold sims inject known γ=±0.02: compute ẽ per galaxy, average,
compare ⟨ẽ⟩ directly to ±0.02 — NO R decomposition. Buildable now for the true-neighbour case with the density
we have + a prior grid; the θ_b integral is the only substantive addition (= the prob-blending model).
NEXT (proposed, not started): prototype the true-neighbour ẽ estimator as a direct test of the posterior formula.

## 2026-07-13 (cont.21) — NOISE-FLOOR study: is the per-bin / global m a real feature-set effect or retrain scatter?

**Concern (user).** The ranked comparisons (conc-v1 vs V1 vs V2, and per-R_blend-bin spreads) may be dominated
by training stochasticity that "rebalances" bias, not by the feature set. The bootstrap ± (±0.20-0.27%) is the
WRONG error bar — it is within-a-fixed-model sampling error, blind to retrain scatter.

**What we know.** Training IS seeded (train_measurement_model.py --seed default 421; torch+numpy+split+shuffle
all seeded), and ALL prior runs used 421 -> init/data-order identical. BUT (1) jobs land on different GPU nodes,
use_deterministic_algorithms off, cuDNN non-reproducible -> 80-epoch chaotic drift, unmeasured; (2) per-R_blend-
bin m is NOT directly supervised (L_response pins response on a flux×size×conc 6×3×5 GRID, not on R_blend
quantiles) -> per-bin is the under-constrained quantity most free to rebalance. So small global diffs
(conc-v1 +0.11 vs V1 +0.03) are almost certainly noise; per-bin spreads are the weakest evidence; only V2's
−1.21% (≈6× larger, coherent, mechanistic) is plausibly real — but UNMEASURED against a noise floor.

**Launched (2026-07-13).** Seed ensemble to measure the floor: retrain the SAME feature set at seeds 422/423/424
and measure scatter of global + per-bin m on the CLEAN 40-139 split only (STEP 2 dropped to halve wall time).
- V2 g0_meas_crowd_conc_full (decision-critical failing variant): train 15041875/79/83 -> val 15041876/80/84.
- conc-v1 g0_crowd_flux_conc (passing baseline): train 15041877/81/85 -> val 15041878/82/86.
job_train_meas_r2.sh gained SEED + SEED_SUFFIX (distinct model paths); new job_validate_step1.sh (40-139 only,
explicit MODEL, optional MP). Decision: a feature-set difference is only real if it EXCEEDS the seed scatter.
If V2's ~−1.2% survives (3 seeds cluster near it) while conc-v1's scatter is ≪1%, the structure-loss effect is
real; if conc-v1 itself scatters ~±0.7%, the whole ranking (incl. round 2) is under-powered and needs ensembling.

## 2026-07-13 (cont.20) — RESULT round 1: measurement noise on size/flux is FREE; dropping true sersic_n+z is the cost. Round 2 launched.

**SPLIT SEMANTICS (clarified 2026-07-13, user Q).** NO constant-gold row trains anything: flow (NLL on
det_meas_crowd_conc_g0.0 + L_response on det_meas_crowd_g0.05, cases 0-199) and R_blend emulator
(lsst_r_extnbr_ho on lsst_sims_fs2_25876/response_catalogue_train.feather) both train ONLY on the main
varying-shear sim; constgold is validation-only. The two sims SHARE case index = same galaxy field (only
shear differs). Flow uses ALL fields 0-199 (no case holdout) -> neither constgold split is field-held-out
for the FLOW. Emulator uses HELDOUT_MIN_CASE=40 (retrain_extnbr.py) -> trained on fields>=40, EXCLUDES
0-39. So "in-train/held-out" is an EMULATOR distinction ONLY: on 40-139 R_blend interpolates (accurate ->
clean FLOW test); on 0-39 R_blend extrapolates (adds emulator error, piles into high-blend q3). Read m gaps
between splits as emulator generalization + case stats, NOT flow generalization. Judge flow realism on 40-139.

**Round-1 validation (constgold, NO correction; labels below = emulator-in-sample 40-139 / emulator-held-out 0-39).**
- **V1/szfl** (measured size+flux; TRUE sersic_n+z kept): STEP1 c40-139 **m=+0.03%±0.20%** ✅, per-R_blend-bin
  spread ISO/q1/q2/q3/q4 = −0.8/+0.2/−3.4/+3.8/+0.1% (~7.2% ≈ 1.07× conc-v1 ✅). STEP2 held-out c0-39
  **m=+0.77%±0.27%** ❌ (spread −0.6/+0.8/−1.8/+6.2/+1.1%). So V1 passes in-train, FAILS held-out.
- **V2/full** (measured size+flux+class_star; DROP sersic_n+z): STEP1 **m=−1.21%±0.20%** ❌, STEP2 held-out
  **m=−0.27%±0.27%**. Per-bin spread ≈ conc-v1 (~6.6/7.7%) → failure is a GLOBAL slope offset, not scatter,
  AND the 0.94% inconsistency between splits is itself disqualifying.

**Diagnosis.** Measurement noise on the shear-EVEN own-props (size, flux) is essentially free (V1 passes).
The −1.21% in V2 comes from DROPPING true sersic_n + redshift; `measured_class_star` alone does NOT replace
them. So the load-bearing information is structural/redshift, and the open question is (a) which of the two,
and (b) whether a fully-MEASURED concentration proxy can recover it (deployability on real single-band data).

**Round 2 launched (2026-07-13).** Three feature sets added to measurement_model.py, all on V2's realistic
base [measured_mag_auto, measured_flux_radius, measured_class_star, nbr_flux_near/far/max] + e1/e2 flow-blind:
- g0_meas_conc_sern (V3a diag): + TRUE sersic_n only. train 15040585 → val 15040586.
- g0_meas_conc_z    (V3b diag): + TRUE redshift only. train 15040587 → val 15040588.
- g0_meas_conc_struct (V4 realistic): + MEASURED mag_aper+fwhm_image+isoarea_image (mean-head forms
  concentration ~ mag_aper−mag_auto). train 15040589 → val 15040590.
New parametrized job jobs/job_train_meas_r2.sh (FS,TAG via --export); validate reuses job_validate_meas.sh.
Decision: if V4 hits |m|≤0.3% both splits + spread ≤1.5× conc-v1 → goal met fully realistically. If only
V3a/V3b recover, the winning axis names what real data must supply (structure meas. vs photo-z).

## 2026-07-12 (cont.19) — NEW DIRECTION: realistic conditioning (measured primary, true neighbours), goal m<=0.3%

**Motivation (user).** conc-v1 conditions the flow on TRUE simulation properties everywhere; real data only
has MEASURED (noisy) properties for the detected PRIMARY source. The SBI object is
`p(ehat, thetahat | e, theta, theta_blending)` — measured observables given true primary latents and true
blending properties. Realism upgrade: move the PRIMARY conditioners from true -> measured observables;
keep neighbours/blending (`theta_blending`) TRUE for now (explicit user directive: "use true neighbors").
Goal: still reach m <= 0.3% under this more realistic conditioning.

**Key design constraints.** (1) The flow TARGET is measured_ngmix_g1/g2, so measured primary SHAPE cannot
be a conditioner (shear-contaminated -> circular response). Intrinsic true e1/e2 has no measured substitute
-> dropped (or mean-head-only via --flow-blind-features, a bridge ablation). (2) Only shear-ROBUST measured
observables (measured flux, measured size) are safe conditioners. (3) Neighbour features nbr_flux_near/far/max
stay TRUE. Documented risk: dropping intrinsic orientation could collapse R_flow the way |e|-only (e_abs)
made recovered shear ~10x too small; open empirical question whether L_response supervision + ensemble
marginalization rescue it.

**Design (workflow wew6xrpj2, 7 agents).** Verify surfaced two decisive facts: (1) dropping intrinsic
e1/e2_input_p is mechanically fatal (R_flow is generated ENTIRELY by reshearing them in the flow-blind
mean head -> R_flow==0 if removed); the correct realistic reading keeps true `e` as the latent shear-map
channel (mean-head-only, never in the density transform; NOT the banned measured shape) and swaps only the
OTHER own-props. (2) Constgold validation catalogues carry NO measured primary observables (40 cols; only
e1/e2_plus/minus + S/N), so a measured-primary lookup must be built from the raw per-case SExtractor+CrossMatch.
Verified the training measured_* cols are DIRECT renames of raw SExtractor MAG_AUTO/FLUX_RADIUS/CLASS_STAR.

**Implemented + LAUNCHED (2026-07-12).** Feature sets registered in sbs_shear/measurement_model.py:
- g0_meas_crowd_conc_szfl (V1 bridge): swap size+flux true->measured (measured_mag_auto, measured_flux_radius),
  keep sersic_n + redshift TRUE; e1/e2 flow-blind; neighbours TRUE.
- g0_meas_crowd_conc_full (V2 realistic): measured_mag_auto + measured_flux_radius + measured_class_star,
  DROP sersic_n + redshift; e1/e2 flow-blind; neighbours TRUE. = p(ehat,thetahat|e,theta,theta_blending).
New: scripts/build_meas_prim_lookup.py (+ jobs/job_meas_prim_lookup.sh) averages measured mag/size/class_star
over +/-0.02 constgold renders per (case,input_index) -> results/meas_prim_lookup_c0-139.feather (shear-even).
validate_constant_with_blend.py gained --meas-prim-lookup (merges measured cols WITHOUT fillna(0.0) -> NaN ->
trained missing-indicator; 95% match gate). Jobs: job_train_meas_szfl.sh, job_train_meas_full.sh (single-change
A/B vs conc-v1), job_validate_meas.sh (parametrized by MTAG). Slurm chain: train 15038070/15038071 + lookup
15038083 -> validate 15038088(V1)/15038089(V2) via afterok. Success bar: |m|<=0.3% BOTH splits AND per-bin
spread not >1.5x conc-v1 (+0.11/-0.07, spread ~6.7/10.7%). NOTHING validated yet — awaiting the chain.

## 2026-07-12 (cont.18) — RESULT: Stage A executed → response is LINEAR, conc-v1 stands (no retrain)

**Ran** `job_resp_target_g02.sh` (job 15037574) → `results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz`,
then `compare_response_targets.py` vs the g=0.05 target. **Verdict: LINEAR within noise where it matters →
Stage B NOT launched; conc-v1 remains the accepted model.**

**Evidence.**
- Global R: **0.2794** (g=0.02) vs **0.2816** (g=0.05) → **−0.78%**, mild and systematic.
- High-response cells (|R|>0.15, 49/90 cells = 53% of galaxies — the ones that actually set R_flow):
  signed `<rel>` = **+0.35%**, `<|rel|>` = 3.5% → **agree within noise, no coherent sign.**
- The `<|rel|>=57%` / `<rel>=+15.8%` headline from the raw script is a **division-by-near-zero artifact**:
  all 52 flagged cells sit in near-zero-R bins (bright/isolated, flux 3–5) where a ~0.013 R-unit noise
  wiggle explodes into ±100–600% rel. Count-weighted **absolute** diff is only 0.0128 R-units.
- The −0.78% global lives entirely in low-response cells (≈0 R_flow contribution) and even **disagrees in
  sign** with the high-response cells (+0.35%) — not the coherent, high-leverage signature a real
  curvature-driven m bias would show.
- Clincher: a coherent ~0.8% target-amplitude bias would push conc-v1's m ~0.5–0.8% one way. It is
  **+0.11%/−0.07%** (straddles zero, ~7× smaller) → the flow does not inherit the target's global
  amplitude; L_NLL + structure regularize it.

**Conclusion.** The target@0.05-vs-validation@0.02 amplitude mismatch is **not** the residual-bias source.
Retraining at g=0.02 would inject ~2.5× more target noise to chase an effect below both conc-v1's margin
and the target-noise floor. Stage B jobs (`job_train_conc_g02tgt.sh`, `job_validate_conc_g02tgt.sh`) and
the optional 200-case extension remain staged-but-unlaunched should the assumption need revisiting.

## 2026-07-12 (cont.17) — STAGED (not launched): g=0.02 response-target amplitude experiment

**Motivation.** The flow's response-supervision target is a secant measured at **g=0.05**
(`Rsim=<[e(0.05)-e(0)]·ĝ>/0.05`, cases 0–99, 6×3×5 flux×size×r_blend cells), but the flow's response
term (`--response-delta 0.02 central`) and the constgold validation both probe **±0.02**. The secant
estimates the slope at ≈0.025, the model/validation at ≈0 — an amplitude mismatch that imprints any
response *curvature* as bias (~1% of R_flow ≈ ~1% of m, i.e. ~10× the current 0.11% margin). Toy scan
says the response is linear to 1–2%, so this is exactly at the level worth checking.

**Preflight finding.** The g=0.02 render (`det_meas_crowd_g0.02_test_full.feather`) covers **only cases
0–99**, not 0–199. So a g=0.02 target is 100-case (not the planned 200) and ~2.5× noisier per galaxy
(smaller signal / same shape noise) — a blind retrain would risk imprinting target noise at λ=300.
Reframed as **confirm-first**.

**Plan (staged, nothing submitted).** Runbook: `jobs/RUN_g02tgt.md`.
- **Stage A (confirm-first, no GPU):** `job_resp_target_g02.sh` builds the g=0.02 target (cases 0–99,
  reusing existing `g0_lookup_c0-99.feather`); `scripts/compare_response_targets.py` diffs it per-cell
  vs the existing g=0.05 target. Two secants-from-0 are equal iff the response is linear over [0,0.05].
  Decision: agree → linear, g=0.05 target already SNR-optimal, **do not retrain**; systematic sign
  (esp. high-r_blend cells) → curvature real, go to Stage B.
- **Stage B (only if curvature):** `job_train_conc_g02tgt.sh` (IDENTICAL to conc-v1 recipe — feature
  set `g0_crowd_flux_conc`, same train catalogue, λ=300, δ=0.02, batch 8192 for a clean A/B — only the
  target changes) → `job_validate_conc_g02tgt.sh` on constgold. Model
  `..._crowdflux_conc_tgt02c99_central02_lam300_v1.pt`. Success = same-or-better global **with smaller
  per-bin spread** (a real amplitude fix flattens, not just rebalances) vs conc-v1's +0.11%/−0.07%,
  spread ~6.7/10.7%.
- **Optional 200-case extension** (`job_g0_lookup_0-199.sh` + regenerate g=0.02 det+meas for 100–199)
  only if Stage B is promising but target-noise-limited.

**Files added:** `jobs/job_resp_target_g02.sh`, `jobs/job_train_conc_g02tgt.sh`,
`jobs/job_validate_conc_g02tgt.sh`, `jobs/job_g0_lookup_0-199.sh` (optional),
`scripts/compare_response_targets.py`, `jobs/RUN_g02tgt.md`. No existing code/model changed.

## 2026-07-12 (cont.16) — FINALIZE conc-v1 as the accepted measurement model; cleanup + review

**Decision.** Adopt `models/measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt`
(feature set `g0_crowd_flux_conc` = own props + `nbr_flux_near/far/max`, absolute response loss,
snc100_central02_lam300) as the accepted constant-gold measurement flow. It reaches sub-0.2% global
multiplicative bias on **independent** gold-constant truth **without** the `deficit(R_blend)` bolt-on:
in-training (40–139) **+0.11%**, emulator-held-out (0–39) **−0.07%**. See cont.15 for the full tables
and the confirm-first orthogonality that justified the `nbr_flux_max` concentration feature.

**Accepted caveats (carried forward, not resolved):**
- Global sub-percent is partly **cancellation**, not per-bin flatness: R_blend-quintile spread ~6.7%
  in-train / ~10.7% held-out (low-blend runs slightly negative, q3 positive). Valid for a
  population-averaged m at this survey's blend mix; a reweighting-sensitivity pass is the recommended
  next check before any paper claim (NOT yet run).
- The reported `±` bars come from a case bootstrap that holds R_flow fixed (validator comment,
  ~line 227) → they understate the true CI.
- q3 residual is a selection + far-pair-measurement + ~0.015 cross-sim mixture (cont.9); both flow and
  emulator are individually verified correct (audit_self_truth / audit_blend_truth), so no single flow
  or emulator knob flattens it — the mixture is why the bolt-on worked empirically.

**Finalization is documentation-only — do NOT flip argparse defaults.** ~8 legacy scripts
(`measure_flow_c*.py`, `fit_additive_correction.py`, `diagnose_additive_origin.py`,
`calibrate_blend_residual_split.py`, `finetune_additive_mean_head.py`, `apply_g0_mean_bias_shift.py`,
`toy_model_calib.py`) hardcode `--measurement-model` default = the NON-conc `crowdflux_lam300_v1.pt`.
They do not supply `nbr_flux_max`; repointing them at conc-v1 would raise `KeyError 'nbr_flux_max'` or
silently drop a conditioning feature. To reproduce/validate conc-v1, pass `--measurement-model` and the
`crowd_flux_conc` lookup explicitly (see `job_validate_conc.sh`).

**Code review (high-effort, 5 deviations/bugs — no crashes in the finalize path):**
1. `build_crowding_lookup.py:67` — `nbr_flux_max` spans the full FAR=7″ (includes the 5–10″ far-pair
   regime cont.9 flagged as a measurement systematic); the model-side comment motivates it as "close
   blend". Intent-vs-implementation deviation to confirm is deliberate.
2. `build_crowding_lookup.py:62` — pair separation is raw Euclidean on (RA,DEC), no cos(DEC); negligible
   at tile dec≈−0.5, silent latent bug if reused off-equator. No dec guard.
3. Finalize hazard above (legacy default models expect the non-conc feature set).
4. `augment_crowding.py` docstring omitted `nbr_flux_max` — **fixed** this entry.
5. `validate_constant_with_blend.py:227` bootstrap holds R_flow fixed → `±` understates the CI.

**Cleanup (executed 2026-07-12, user-approved).** No git in this tree, so all moves are reversible
archives, not deletes — except the one binary the user OK'd removing.
- Retracted/regressed conc experiments → `jobs/archive/`: `job_train_conc_relerr.sh`,
  `job_validate_relerr.sh` (relerr retrain regressed: q3 +9.3%, global +1.05%), `job_val_conc_fullemu.sh`
  (confounded — its 0–39 blend-lookup rows are a different emulator build).
- **Deleted** the redundant regressed model
  `measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_relerr_lam300_v1.pt` (+curve, ~5.5 MB).
- Archived the Jul 8–10 dead-end diagnostic batch: **27 scripts → `archive/`** (`combiner_*`,
  `analytical_*`, `derived_fit*`, `rblend_nonlin`, `robust_nonlin`, `dip_vs_shear`,
  `render_diff_mechanism`, `emu_domain_shift`, `perobj_analysis`, `q3_outlier_check`, etc.) and the
  **21 companion jobs → `jobs/archive/`** that referenced them (identified deterministically; verified no
  keeper caught, no dangling reference remains).
- **Kept** (grounding standing conclusions / production / separate project): the 4 production scripts,
  `audit_blend_truth`, `audit_self_truth`, `toy_shear_scan`, `build_blend_multiplicity`,
  `match_fixed_sample`, `map_truth_cases`, the `probblend_*` set, and the conc/qdiag/audit-truth jobs.
- Post-cleanup: `scripts/` 48 `.py`, `jobs/` 128 `.sh`; `archive/` 70, `jobs/archive/` 81.

**Files changed this entry:** `augment_crowding.py` (docstring), `WORKLOG.md`. No model or validator
logic changed.

## 2026-07-11 (cont.15) — bolt-on curve is regime-bound; root-cause feature identified (confirm-first)

Two diagnostics probing whether the cont.14 `deficit(R_blend)` recalibration is a *robust* fix or a
regime-bound patch, and whether the flow/emulator can fix q3 at the source instead.

**1. Portability to the emulator's held-out set (`job_qdiag_holdout.sh`, job 15028865).** Apply the
40–59-fit curve to cases **0–39** (the extnbrho emulator's TRUE held-out set — excluded from its
training). Before→after:

| set | uncorrected | corrected | q3 (corr) |
|---|---|---|---|
| in-training (40–139) | +0.81% | **+0.28%** | +0.6% |
| **emulator-held-out (0–39)** | **+1.63% ± 0.27%** | **+1.16% ± 0.27%** | **+3.7%** |

The curve helps on held-out (+1.63→+1.16, ~29%) but **does not reach Stage-IV**, and q3 barely moves
(+3.7% vs +0.6% in-training). ⇒ the post-hoc calibration **inherits the emulator's blind spots**: it
works where the emulator is accurate, only partially transfers where it is not. Not a robust fix.

**2. Confirm-first orthogonality — would a flow retrain actually help? (`job_qdiag_ortho.sh`, job
15029001, cases 40–79; new validator block: deficit by `nbr_flux × n_pairs`).** The flow's ONLY
crowding input is `nbr_flux_near/far`. Bin the blended set into tight `nbr_flux` quartiles (fix what the
flow sees), split each by `n_pairs`, measure deficit `R_sim−R_flow−R_blend` at MATCHED `<flux>`:

| flux Q | deficit (low n_pairs) | deficit (high n_pairs) | Δ(nhi−nlo) |
|---|---|---|---|
| Q1 | −0.0200 | −0.0271 | −0.0071 |
| Q2 | +0.0006 | −0.0065 | −0.0071 |
| Q3 | **+0.0257** | +0.0133 | **−0.0124** |
| Q4 | +0.0253 | (split degenerate at n_pairs cap=20) | — |

Three quartiles, **same sign, monotonic in flux (~2–2.5σ)** ⇒ at fixed flow-input the deficit still
depends on multiplicity → `n_pairs` carries information the flow structurally cannot access. **Sign flips
the mechanism**: the large +deficit lives in LOW-`n_pairs` cells (flux **concentrated** in one bright
close blend); spreading the same flux over many faint neighbours shrinks it. So the flow needs a flux
**concentration/dominance** feature (`rb_max/nbr_flux` or `n_pairs`), NOT raw count. (Reconciles with
cont.14's "dominance-flat at fixed R_blend": controlling for `nbr_flux` instead of `R_blend` flips the
sign — concentration is the true driver.) Bonus: binning by bright-OOD flux shows m going negative
(−3.2%, −4.7%) — a SEPARATE *emulator* over-prediction, not a flow issue.

**Conclusion.** Recommended root-cause fix: retrain `measurement_flow_...snc100_central02_lam300` adding
a flux concentration/dominance feature to its conditioning, then re-validate on constgold. Confirm-first
is positive; awaiting go-ahead to launch the retrain.

**RETRAIN RESULT (2026-07-11).** Added `nbr_flux_max` (log-scaled brightest-single-neighbour flux within
7", pure truth geometry) as a new conditioning feature -> feature set `g0_crowd_flux_conc` (near+far+max);
model `measurement_flow_g0_ngmix_crowdflux_conc_snc100_central02_lam300_v1.pt`. Pipeline jobs
`job_crowd_conc_prep.sh` (15029825) -> `job_train_crowd_conc.sh` (15029826, 80 ep, val logp -2.163) ->
`job_validate_conc.sh` (15029828). Constgold 100-case (40-139), **NO deficit correction**:

| bin | baseline (no corr) | bolt-on (in-train) | **conc flow (no corr)** |
|---|---|---|---|
| ISO | ~-0.7% | ~0% | -2.6% |
| q1 | ~-1% | +0.1% | -2.3% |
| q2 | +1.9% | +0.9% | -1.1% |
| q3 | +6.8% | +0.6% | +4.1% |
| q4 | +0.1% | -0.3% | -0.0% |
| **global** | **+0.81%** | +0.28% | **+0.11%** |

The concentration feature cut the 100-case global to **+0.11% with zero post-hoc calibration** (beats the
bolt-on's +0.28%), directly resolving the "don't want an extra calibration model" concern. q3 dropped
+6.8%->+4.1% (partial, not fully flat); a mild negative tilt appeared at low blend so the excellent global
is partly cancellation. Per-bin flatness is worse than the in-train bolt-on, but the bolt-on's weakness was
never in-train flatness — it was PORTABILITY.

**Held-out 0-39 (absolute-conc, job 15029828):** global **-0.07%** (baseline +1.63%, bolt-on +1.16%) —
root-cause feature PORTS where the bolt-on failed. BUT held-out quintiles ISO -2.7%, q1 +2.2%, q2 +0.8%,
q3 **+8.0%**, q4 +0.5% (spread ~10.7%): the -0.07% is CANCELLATION, and held-out q3 (+8.0%) > in-train q3
(+4.1%). **Decomposition**: residual = (a) flow self-response deficit [flow-fixable, seen in-train] + (b)
EMULATOR held-out generalization [0-39 excluded from emulator -> R_blend inaccurate -> inflates held-out q3;
NOT flow-fixable]. Flow retraining can flatten in-train at best; robust held-out needs a better emulator.

**ITERATION 2 — relative response error (job 15034993/94): REGRESSION, reverted.** conc feature +
`--response-error relative`. In-train (40-139) profile ISO -2.1%, q1 +1.7%, q2 +3.0%, q3 **+9.3%**, q4
+0.9%, global +1.05% (spread ~11.4%). Relative error UP-weights the small-response faint/crowded tail and
DE-weights the moderate-response q3 region -> lowered R_flow at q3 (0.185->0.166) -> q3 deficit WORSE.
Wrong lever. **Best flow remains the absolute-loss conc v1** (`..._crowdflux_conc_snc100_central02_lam300_v1`).

**GROUNDING (audit_blend_truth.py, job 15037045): EMULATOR EXONERATED.** Per-pair emulator R_blend vs
sim-measured truth (delta_et1/gamma) on gold 0-39, IDENTICAL pairs (section A): ratio pred/truth ~1.00
across ALL neighbour-mag and distance bins (diffs ~0.001). The emulator is accurate per-pair. The audit's
(B) per-primary table showed a spurious truth-emu=-0.097 at q3 ONLY because its own sanity check failed
(<pred_sum>=0.084 over ~8 stored pairs vs production <R_blend>=0.168 over k=20 -> pair-COUNT mismatch, not
emulator error). ⇒ the in-training q3 deficit is NOT emulator; it is FLOW or coherent anisotropy. This
CONFIRMS the earlier multiplicity-discriminator "q3=flow" by an independent method and refutes the audit's
own "emulator under-predicts +0.047" hypothesis. The flow is the correct lever (conc-v1 q3 6.8->4.1 proves
flow work moves it). Also: the earlier "held-out=emulator-limited" claim and the +8.29% full-emulator test
are RETRACTED (confounded: blend_lookup_c0-199's 0-39 rows come from a different emulator build; the "ext"
builder only covers 40-199). Response target is r_blend-resolved (5 bins) and NOT double-counting
(ISO R_flow~=R_sim), so the flow WAS supervised per blend bin yet still under-fits q3 -> a capacity /
additive-cross-term / anisotropy limit, not missing supervision. OPEN: is residual +4.1% q3 flow-underfit
(targeted retrain helps) or coherent anisotropy (isotropic emulator+flow can't capture -> needs angle-aware
blend term)? Diagnose before another retrain.

**RESOLVED (audit_self_truth.py job 15037094 + reconciled with cont.8/cont.9): q3 deficit is a
SELECTION+MEASUREMENT MIXTURE, NOT flow and NOT per-object nonlinearity.** True self-response R_self_truth
vs model R_flow per R_blend quantile, gold 0-39: q3 R_flow=0.175 ~= R_self_truth=0.173 (flow-truth=+0.003)
=> **the flow correctly predicts the self-response at q3** (ISO flow over +0.026, q1 under -0.029, q4 over
+0.014 -> minor). Emulator ALSO accurate (audit_blend). So neither component is individually wrong at q3.
CAUTION (self-corrected after user challenge): the residual R_sim(0.42) > R_flow+R_blend(0.39) is **NOT a
super-additive nonlinear cross-term** — cont.8 TOY (toy_shear_scan.py) showed the per-object blend response
is LINEAR in shear (flat to 1-2%; detects nonlinearity where it exists via the +19% self-response row), and
cont.9 fixed-sample showed the deficit is a MIXTURE: ~1/3 genuine selection + a far-separation (5-10")
weak-pair measurement systematic on near-zero signal + cross-sim offsets (~0.015 at ISO). Per-object R_self
and R_blend DO add linearly; the ensemble gap is a SAMPLE/MEASUREMENT mismatch, not physical nonlinearity.
**This OVERTURNS the cont.14 "q3=flow over-suppresses" conclusion** (multiplicity discriminator inferred
flow from deficit STRUCTURE; direct truth shows flow is fine), and my own transient "nonlinear cross-term"
label is RETRACTED. conc-v1's q3 gain (6.8->4.1%) was a HACK (R_flow over-predicts 0.185 vs truth 0.173 to
mask the deficit -> its ISO tilt/spread). **IMPLICATION (matches cont.9): the deficit(R_blend) bolt-on is
the right PRACTICAL fix because it empirically absorbs the mixture; no single principled knob (flow or
emulator) addresses it since both are individually correct.** Next: keep the bolt-on; robustness lever =
make deficit(R_blend) portable (original cont.14 caveat), NOT more flow retraining. Flow-side lever near its limit; held-out q3 is
emulator-limited. Open forks: (i) accept conc-v1 as the flow result; (ii) add n_pairs as an explicit
feature (ortho showed nbr_flux_max did NOT absorb the fixed-flux n_pairs structure); (iii) improve the
emulator's held-out generalization. Speed note: flow training GPU-STARVED at batch 8192 on 2080 Ti
(~25% util); batch 16384+ helps.

## 2026-07-10 (cont.14) — SUB-PERCENT REACHED: deficit(R_blend) recalibration (autonomous /loop)

**Goal (user /loop).** Read last run, reason, devise tests; loop until sub-percent constgold `m` is
found or judged infeasible under the framework.

**Starting point.** 100-case constgold (40–139) `WITH blend m = +0.81% ± 0.20%`; residual concentrated
in the q3 R_blend quintile (**+6.8%**, reproducible across independent case halves +6.4%/+6.9%), while
ISO/q1/q2/q4 sit within ±1.5%. Global sub-percent only by cancellation → fragile.

**Mechanism diagnostic (`job_qdiag_mult.sh` → `build_blend_multiplicity.py` c40–79 + validator
multiplicity discriminator, job 15023009).** Split each R_blend quintile by n_pairs and by dominance
(one neighbour carries >70% of |R_blend| vs many-small). Absolute deficit `R_sim−R_flow−R_blend`:
- **q3 is dominance-FLAT** (1-dominant 0.0238 vs many-small 0.0250) but grows with n_pairs
  (0.0196→0.0305), and R_flow drops faster than R_sim with crowding ⇒ **q3 = the FLOW over-suppressing
  the self-response at moderate crowding**, NOT emulator super-additivity.
- **q4 is dominance-DRIVEN** (1-dominant m=+10.1% vs many-small m=−29.7%, R_blend=0.909 ≫ true 0.63)
  ⇒ emulator additive sum is **sub-additive/saturating** for many overlapping bright neighbours; the
  two halves cancel to +0.1% globally.
- `deficit(R_blend)` is **non-monotonic**: ISO −0.008, q1 −0.003, q2 +0.004, **q3 +0.025**, q4 ~0 — a
  clean bump at q3, with a large negative spike (−0.15) only in the extreme top bin (R_bl≈1.18).

**Fix — empirical `deficit(R_blend)` recalibration (`--fit/apply-rblend-corr`, `job_qdiag_deficit.sh`,
job 15025379).** Fit 24-bin `deficit(R_blend)` on cases **40–59**, apply **out-of-sample** to held-out
**60–79**. Same-binning before/after:

| bin | uncorrected (control) | corrected (OOS) |
|---|---|---|
| ISO | −0.7% | +0.0% |
| q1 | −1.1% | +0.1% |
| q2 | +1.9% | +0.9% |
| **q3** | **+5.0%** | **+0.2%** |
| q4 | +0.7% | −0.3% |
| **GLOBAL** | **+0.91% ± 0.35%** | **−0.19% ± 0.35%** |

The q3 bump collapses on cases the correction never saw ⇒ the residual is a **transferable function of
blend strength**, not covariate-shift noise. (Bug caught & fixed mid-run: default `--max-rows` truncated
the case-ordered read before the `--min-case` filter → garbage N~70k; fixed with `--max-rows 40000000`.)

**Headline (`job_qdiag_corr100.sh`, job 15026858).** Correction (fit on 40–59) applied to the FULL
100-case set (40–139; 80% out-of-sample): **`m = +0.81% → +0.28% ± 0.20%`** — **sub-percent, under the
Stage-IV 0.3% target.** (Per-bin q3 confirmation at 100-case scale running.)

**CONCLUSION.** Sub-percent constgold bias **is reachable within the current framework** via an empirical
`deficit(R_blend)` response recalibration that **transfers out-of-sample across independent case
realizations**. Mechanistically it corrects the flow's moderate-crowding self-response over-suppression
(the q3 bump) plus the emulator's extreme-blend saturation.

**Open caveat (for real-data applicability).** The correction is currently fit on constgold and applied
to constgold (cross-case but same simulation). The real-data-portable version must derive
`deficit(R_blend)` from the MAIN training set's per-object truth (not from the validation set itself).
Recommended next test: fit on main-set responses, apply to constgold.

**Artifacts.** `results/deficit_rblend_c40-59.npz` (correction curve),
`results/blend_multiplicity_extnbrho_c40-79.feather`, jobs `job_qdiag_{mult,deficit,corr100}.sh`.

## 2026-07-09 (cont.13) — NEW DIRECTION: probabilistic blending (forward-model the neighbour distribution)

**Motivation (user).** The framework currently feeds `R_blend` the TRUE neighbour field
(`predict_response(field, field)` in `build_blend_lookup.py`). In real data we only have the
DETECTED catalogue with MEASURED properties; the neighbour population must be forward-modelled,
and its undetected part is set by the classifier (selection informed by blending). Design in
`PROB_BLENDING.md`.

**Framework (PROB_BLENDING.md).** Marginalise `R_blend(ô_i) = E_{N~p(N|ô_i)}[Σ f_reg]`. Because
`f_reg` is additive, split neighbours into DETECTED (in the catalogue → directly summed) and
UNDETECTED (forward-modelled). The undetected term is an intensity integral (Campbell):
`R_blend^undet = ∫2πθ dθ ∫dm ds dn  Φ(m,s,n)·[1−p_det]·f_reg`, where `Φ` is the parent property
function and `[1−p_det]` is the **already-trained classifier** as the thinning kernel. No emulator
retraining; the only new ingredient is the population prior `Φ`.

**Characterisation (`scripts/probblend_characterize.py`, cases 0–7, tag `lsst_r_extnbr_ho`).**
Decomposed truth-fed `R_blend` over 2.26M detected primaries into detected- vs undetected-neighbour
parts:
- `<R_blend full> = 0.1931`; **undetected neighbours carry 7.8% (`<undet>=0.0151`)**.
- **Dropping undetected neighbours ⇒ Δm ≈ +3.28%** (`<undet>/R_total`, R_total=0.462) — a *dominant*
  systematic, far larger than the ~1.7% realization-variance residual chased in cont.1–12. So this
  is the real target for Stage-IV |m|<0.3%.
- Undetected fraction is largest for faint primaries (r_p 27–28: 13.6%) and for bright primaries
  (r_p 18–23: 22.4%, many faint undetected neighbours around bright galaxies).
- The undetected signal lives at secondary magnitude **r_s ≈ 26.5–28** (detected fraction there
  falls 40%→1%). Bright r_s<24 neighbours (65% of R_blend) are ~96% detected → easy.

**Forward model RESULT (`scripts/probblend_forward.py`, conditioned production model).** Level A:
classifier reproduces the hard detection-truth undetected census to 8% (`undet_soft/undet_hard=1.084`).
Level B (population integral, NO true neighbour positions):
- ξ(θ)≈1 measured — the sim field is Poisson (random placement); clustering is NOT the residual.
- Three ingredients, all essential: (1) field-LF neighbour draw + area-uniform separations; (2)
  icat2cla-faithful detection context for the neighbour (nearest of {primary@θ, Poisson field
  gal@d_f}, isolated>3″) — fixes θ>2″; (3) conditioning weight `p_det(primary|nbr)/p_det(primary|iso)`
  — down-weights bright close neighbours that would kill the primary's detection — fixes θ<1.5″.
  Diagnosed per-θ-shell in `probblend_ctx_diag.py`: primary-only ctx 0.82×, naive field ctx 1.31×,
  conditioned 1.03× soft.
- **Bottom line: dropping undetected nbrs = Δm +3.29% → conditioned forward model = Δm −0.37%**
  (geometry −0.10%, classifier calibration −0.28%). ~9× reduction, near Stage-IV target.
- Per-primary-magnitude residual tilts +1.4%(r_p26-27)/−2%(r_p27-28) — faint-primary end is the weak
  point; partly cancels in the global.
- **Classifier recalibration (`probblend_calib.py`) closes the calibration piece:** the classifier is
  under-confident in the dominant response bin (p_det 0.9-1.0 = 87% of weight; predicts 0.974, actual
  0.986). Isotonic recalibration on detection truth drives soft/hard 1.084→0.999, i.e. classifier
  Δm +0.28%→-0.00%. **Final global ladder: drop-undetected +3.29% → forward model -0.37% →
  +recalibration ≈ -0.10%.** Sub-0.3% achieved with no true neighbour positions.
- **Per-magnitude tilt diagnosed = context-dependent classifier miscalibration** (not the conditioning
  weight, which is ⟨0.97⟩ everywhere). soft/hard flips sign with primary mag: 1.18× (over) at r_p 25-26
  → 0.65× (UNDER by 35%) at r_p 27-28 (faintest, ⟨p_det_iso⟩=0.10, near limit). Forward model
  reproduces soft so inherits it. **Tested (`probblend_calib2d.py`): NEITHER global nor
  magnitude-conditional (2D) isotonic calibration fixes the faint bin** — r_p 27-28 stays soft/hard
  0.74 either way (both fix only the global → 1.00; worst per-bin ~26% unchanged). So the faint-primary
  tilt is a STRUCTURAL limit of the pairwise classifier (a neighbour of a detected-faint primary is in
  a biased hard-to-deblend config the single-pair model can't see), not a calibration artefact. Global
  impact small (~3% of pairs → global m ~0.1%); per-bin robustness needs a richer-context detection
  model. User's next step.
Full writeup + caveats in `PROB_BLENDING.md` §8.

## 2026-07-09 (cont.12) — CONCLUSION: render is BIT-REPRODUCIBLE; the +1.6% is finite-SAMPLE variance, not a bug

DECISIVE: re-rendered cases 0-4 (same seeds 123-127, current code, REPRO dir) produce BIT-IDENTICAL images
to the originals (md5 case0 d39f85d4...==d39f85d4, case1 b78993f3...==b78993f3, same 1.32GB). So the pipeline
is fully DETERMINISTIC. (Killed job 15011091 after the image test — no need for shape/response.)
Noise: single constant rmsExpo_r=0.3115 keyed by TILE not case -> identical noise level all cases.
=> code, env, galaxy DISTRIBUTIONS (Re/n/q/mag to 0.1%), noise level, AND the render are ALL identical
between the 0-39 and 40-79 batches. The ONLY difference is the seed-dependent REALIZATION (which specific
galaxies drawn + which noise pattern).

THEREFORE the +1.6%(0-39) vs +0.15%(40-79) difference is pure FINITE-SAMPLE realization variance -- NOT a
framework bug, NOT selection/nonlinearity/dip/emulator. The naive per-case bootstrap (+/-0.27%) UNDER-
estimates the true batch-to-batch uncertainty because cases within a batch share ~50% of their galaxies
(case0 cap case40 = 50.4% RA overlap; independent draws from the same 1.8M master) -> strongly correlated ->
effective N << 40 -> true SE ~0.5-1%. So +1.6% is only ~2sigma, not a confident detection.

SESSION BOTTOM LINE: the framework (R_flow + R_blend) gives +0.15% on a fresh independent batch (goldxfer
40-79 baseline, no correction). The 0-39 +1.6% is within the sample/realization uncertainty. The entire
mechanism hunt (dip / shear-nonlinearity / selection / empirical dR(R_blend) / emulator retrain) was chasing
a bias that is NOT robustly significant at the 0.3% level. The empirical dR correction "worked" only because
it fit 0-39's specific realization; it over-corrects on 40-79.
RECOMMENDATION to reach a <0.3% VALIDATION: (1) much larger independent sim volume, and crucially decorrelate
fields (draw NON-overlapping galaxy subsets per case) so the bootstrap is valid; (2) measure the bias across
several independent batches to get the true SE; (3) only then interpret any residual as framework bias.

## 2026-07-09 (cont.11) — CORRECTIONS: morphology is SERSIC (not generative); render diff is ~1.5%/3.9sigma

Two corrections to cont.10:
1) MORPHOLOGY: the render uses SERSIC everywhere. Import chain: ImSimSkySimple.py `import ImSimObject as
   ObjModule` -> ImSimObject.py = bulge+disk galsim.Sersic ONLY (no image loading). The generative module
   ImSimObject-gen.py (SampledGalaxies/InterpolatedImage, genmodel 43.6%) is NEVER imported -- a dead WIP
   file that only looked active because it had uncommitted git edits. My generative-morphology thread was a
   RED HERRING. Config: survey=one_tile, image_type=simple.
2) MAGNITUDE: my ad-hoc r_sim reconstruction skipped the validator's load() selection cut
   (source_select_selection) -> I compared FILTERED 0-39 (cg_dump) vs UNFILTERED 40-79 (recon), inflating
   the gap to 3.2%/8.6sigma. The consistent VALIDATOR numbers: 0-39 m=+1.6%+/-0.27%, 40-79 +0.15%+/-0.26%
   -> diff 1.45%+/-0.37% = **~1.5% / 3.9sigma**. Still a significant systematic, but smaller; the
   "lower at every mag" mechanism plot was partly the filter mismatch.

IMPLICATION: Sersic is DETERMINISTIC from (Re,n,q,mag) labels, which are identical between batches (0.1%).
So at fixed mag the two batches SHOULD give identical mean response -> a 1.5% offset shouldn't occur for a
Sersic render w/ matched inputs -> pushes toward seed/noise realization or field-sample variance BIGGER than
the bootstrap's +/-0.27% (i.e. bias more sample-limited than the bootstrap implies). Repro re-render (cases
0-4, SAME seeds 123-127, current code, scratch REPRO dir; job 15011091) is the decider: Sersic+same seed
should be ~bit-reproducible vs originals -> match => seed/sample variance; differ => pipeline non-determinism.

## 2026-07-09 (cont.10) — THE BIAS IS RENDER-DEPENDENT (systematic, 8.6sigma): 0-39 vs 40-79 differ

GOLD-STANDARD held-out 40-79 (job 15007814 goldxfer): BASELINE (no correction) m = **+0.15% +/- 0.26%** --
the framework hits target OUT OF THE BOX on freshly-rendered fields, while 0-39 gave +1.6%. All lookups
matched 100%; entire diff is in R_sim (measured), even for ISOLATED objects (global shape-response offset).

Provenance: renderer code (ImSimObject-gen.py, sex config: 2025 mtimes), input galaxies (SampledGalaxies/
SimInputCatalog: 2024), shape code (run_shape/shape.py: Apr 2026) ALL identical between renders. Only
blendemu INFERENCE changed (Jul 4: "use trained selection cuts/aperture from metadata") -- affects R_blend
(identical for both) not R_sim. 0-39 rendered Jun 28-29; 40-79 rendered Jul 9 (this session's xval job).

DECISIVE stat-vs-systematic test (compare_render_rsim.py job 15010998): per-case <r_sim> distributions:
  0-39 <r_sim>=0.4698 (range .455-.490) ; 40-79 =0.4549 (range .441-.470). Diff +0.0149 = **+8.6sigma**
  (Welch p=7e-13, KS p=2e-10). Distributions BARELY overlap. Bootstrap SE (0.27%) is CORRECT.
=> SYSTEMATIC render difference, NOT statistical. 40-79 has ~3% lower coherent response AND ~11% MORE
objects/field (298k vs 269k) -> hint: DETECTION detected more (faint) objects -> SELECTION shift.
So the measured constgold bias is RENDER-DEPENDENT by ~1.5-3% (even flips sign +1.7% vs -1.5%), >> the
0.3% target. This reframes the ENTIRE session: the +1.6% we chased is partly a property of the 0-39
detection/render, not a stable framework bug. Running render_diff_mechanism.py (job 15011008): R_sim vs
magnitude + reweight to isolate SELECTION (detection mag-mix) vs RENDERING (response at fixed mag).
NOTE: cmprend 40-79 r_sim recon (0.4549) is ~1.5% below the validator's (0.4623) -- small recon/selection
mismatch to reconcile, but the systematic 0-39>40-79 conclusion is robust (both agree).

## 2026-07-09 (cont.9) — FIXED-SAMPLE test: selection is ~1/3, NOT the whole story; deficit is a MIXTURE

Matched the pairs present in BOTH the g=0.02 and g=0.2 response catalogues (same case+primary+secondary
position; match_fixed_sample.py job 15010596) to remove the shear-dependent sample and isolate the per-pair
response. Result (5%-trimmed):
  FULL ratio(0.02/0.2)=1.112 ; MATCHED(fixed)=1.075 ; drop-in/out pairs = 1.0% (233k/22.7M).
=> SELECTION accounts for only ~1/3 of the 11% (1.112->1.075); ~2/3 PERSISTS on identical pairs.
So "it's selection" (cont.8) was OVERCLAIMED - selection is a ~1/3 contributor, not the driver.
Matched by separation: 0-1" 0.989, 1-2" 1.048, 2-3" 0.937, 3-5" 1.009, **5-10" 1.117 (17M pairs)**.
The physically-important CLOSE pairs (0-3") scatter ~1.0 (+/-5%, consistent w/ toy linearity); the
persistent 7.5% is DOMINATED by the far bin (5-10") where response~0.002 (~0) and the 1.12 ratio is a
0.0002 absolute diff -> looks like a cross-match/measurement SYSTEMATIC on near-zero signal, not a real
blend response, but the huge pair count still moves the summed R_blend.
HONEST BOTTOM LINE: the +1.6% is a MIXTURE of small (~few-%) sources - modest genuine selection (~1/3),
a far-separation weak-pair systematic, close-pair response ~linear. No single clean mechanism -> why the
empirical ΔR(R_blend) correction (absorbs all) reaches subpercent while no single principled knob does.

## 2026-07-09 (cont.8) — TOY: per-object blend response is LINEAR -> deficit is likely a SELECTION effect

User hypothesis: the full-sim g-dependence is a SELECTION effect (detected/cross-matched sample shifts with
shear), not intrinsic per-object nonlinearity. Toy discriminator (toy_shear_scan.py, single controlled pair,
ALWAYS present, no detection gate, ngmix antithetic +/-g; FIXED to working params flux=1000 hlr=0.4 after a
first run railed with hlr=0.3<PSF):
  BLEND response delta_et/g vs g (0.02->0.2): sep 1.2" 1.236->1.221, 2.0" 1.403->1.377, 3.0" 0.067->0.0655
  = FLAT to ~1-2% (LINEAR). Not a null-detector: the SELF-response sanity row IS flat ~2.6 for g<=0.1 then
  +19% at g=0.2, so the toy detects nonlinearity where it exists.
=> per-object BLEND response is ~linear in g -> the ~11% g-dependence measured in the FULL sims (robnl) is
NOT per-object nonlinearity -> most likely SELECTION (population change), which also explains why each piece
"checks out" yet the sum falls short and the net R_blend g-dependence was ambiguous.
Caveats: toy self-response reads ~2.6 not ~1 (convention/normalization offset; does NOT affect the flatness
conclusion, a ratio); toy is idealized (Gaussian, single pair, no detection threshold) -> establishes
per-object nonlinearity is NOT the driver, does not itself PROVE selection.
DIRECTION SHIFT: retraining at small g likely won't help (per-pair response already right). Lever = SELECTION:
model the selection/detection response (metacal-style) or match emulator training selection to constgold;
"improve emulator resolution/features" fits here. NEXT (proposed): isolate the selection response in the full
sims. goldxfer (empirical transfer on fresh 40-79) still running as the practical baseline.

## 2026-07-09 (cont.7) — Dip = shear-magnitude nonlinearity (separation-dependent, competing signs)

Built the ISOTROPIC blend response at g=0.02 (retrieve_response shear_cases=['0.0','0.02'], all 200 fields
have the 0.02 render; resp002_c40-59.feather) to compare against g=0.2. The plain mean is OUTLIER-dominated
(delta_et/0.02 heavy tails; field-pairing does NOT help -> not field variance). Robust 5%-TRIMMED mean
(robnl, job 15010328) resolves it:
  <resp>(0.02)=0.0089 vs (0.20)=0.0080  -> +11% at weak shear, +3sigma (RIGHT sign for the deficit).
  By separation (trimmed ratio 0.02/0.20): 0-1" 1.01, **1-2" 1.09**, 2-3" 0.96, 3-5" 1.04, 5-10" 1.17.
BUT the FULL mean has the OPPOSITE sign (per-primary R_blend 0.02=0.076 < 0.20=0.085) because CLOSE bright
pairs give genuinely LARGER response at large shear (dip test 0-0.4": g=0.2=0.122 vs g=0.02=0.047, real not
noise) and dominate the sum. So the nonlinearity is SEPARATION-DEPENDENT with COMPETING signs:
  - mid-sep (~1-2", near the dip): weak-shear response larger -> emulator(0.2) UNDER-predicts (deficit dir).
  - close pairs: large-shear response much larger -> emulator(0.2) OVER-predicts for constgold.
These partially cancel -> net R_blend nonlinearity sign is ambiguous at this noise -> why no single knob
closes it and the empirical recal works. Principled fix (retrain emulator at g=0.02) would fix the dominant
mid-sep piece. DEFINITIVE resolution = build blend response at g=0.05 (2.5x less noise) via run_shape
--shear_case=0.05 --targets primaries (0.05 image exists, secondaries-sheared = blend config; only primary
SHAPES missing) then 3-point full-mean response(g) per separation. [Not yet launched - bigger MPI shape job.]

## 2026-07-09 (cont.6) — WITHIN-SAMPLE TRANSFER SUCCEEDS: corrected held-out m = -0.29% (SUBPERCENT)

applyhalf (job 15005759): fit deficit(R_blend) on cases 0-19 -> apply rblend_corr_c0-19.npz to HELD-OUT
cases 20-39 (N=5.38M, read pre-overwrite 0-39 catalogue). RESULT:
  GLOBAL corrected m = **-0.29% +/- 0.40%** (baseline ~+1.76%). q3 +9% -> -1.8%. All R_blend bins within
  ~2% (ISO -0.3, q1 -1.4, q2 +1.0, q3 -1.8, q4 +0.2). Two case-halves of 20-39 both concordant.
=> The empirical blend-strength recalibration TRANSFERS to held-out cases and reaches the 0.3% target.
FEASIBILITY DEMONSTRATED (within-sample). Caveats: (1) case-split, not independent fields -> goldxfer on
fresh 40-79 (job 15007814, ~running) is the stronger test; (2) +/-0.40% -> consistent with 0 but not
tightly pinned, more cases would shrink it; (3) empirical correction absorbs the coherent-vs-isotropic /
shear-magnitude gap without a first-principles model (angle-aware/small-g emulator = the principled fix).

## 2026-07-09 (cont.5) — Held-out 40-79 rendered; GOLD-STANDARD transfer test launched

Dip test (job 15005785) CONFIRMED the user's lead: isotropic (half-shear truth + emulator) blend response
vs separation has a DEEP DIP at ~1" (0.117 at contact -> 0.024 at 0.8-1.2" -> 0.08 plateau at 2-3.5"),
faithfully reproduced by the emulator. COHERENT (constgold single-dominant) blend stays elevated and
EXCEEDS the emulator by +0.06-0.09 at close sep (<0.8"). => the dip is an angular-averaging feature the
coherent alignment washes out; emulator (isotropic, no angle feature) under-predicts the coherent close-pair
response. CAVEAT: pure isotropic averaging over isotropic neighbour positions shouldn't create a NET gap
(symmetry) -> the amplitude driver is likely the shear-magnitude mismatch (emulator g=0.2 vs constgold 0.02;
self-response shows 3.7% saturation at 0.05) or the both-sheared cross-term. FIX candidate: angle-aware +
small-g R_blend retrain (angles are stored in the response catalogue, currently discarded).

Correction-curve stability preview: rblend_corr fit on cases 0-19 vs 0-39 agree in SHAPE (rise to +0.05-0.08
at high R_bl, dip -0.15 at extreme), per-bin RMS 0.0096 vs mean|deficit| 0.025 -> transferable in shape,
borderline on 0.3% precision; a SMOOTHER curve (fewer bins) is the ready adjustment if needed.

**40-79 renders COMPLETED** (job 15000850, 3.8h). NOTE: the xval merge OVERWROTE
constant_response_catalogue_train.feather -> it now holds ONLY cases 40-79 (11.9M rows); the 0-39 merged
catalogue is gone (re-mergeable from case0-39 dirs if needed; 0-39 lookups + correction curves already saved).
Gold-standard transfer test: job 15007813 builds blend_lookup_extnbrho_c40-79 + ood_split_c40-79
(crowd_flux_c0-199 already covers 40-79), then job 15007814 runs BASELINE + WITH-correction
(rblend_corr_c0-39.npz) on held-out 40-79. Plus applyhalf (fit 0-19 -> apply 20-39, job 15005759) for an
early within-sample number. <0.3% out-of-sample => feasible; else smooth curve / angle-aware retrain.

## 2026-07-09 (cont.4) — ATTRIBUTION COMPLETE: deficit = coherent(small-g) minus isotropic(large-g) blend

Self audit (job 15004817): R_flow MATCHES R_self_truth in q3 (0.1754 vs 0.1725) -> BRANCH A REJECTED too.
Full truth decomposition (coherent_blend = R_sim - R_self_truth vs emulator isotropic R_blend):
  q1 0.008 vs 0.034 (-0.026); q2 0.091 vs 0.082 (+0.008); q3 0.258 vs 0.219 (+0.039); q4 0.782 vs 0.757
  (+0.025). NOT a uniform nonlinear scale (would break q4/q1).
=> Both models are individually ACCURATE vs their truth. The deficit is definitionally the gap between
what the emulator provides (ISOTROPIC blend at large g=0.2, no angle feature) and what constgold needs
(COHERENT blend at weak g=0.02). Two discarded dependencies: coherence (angle) and shear-magnitude.
Evidence of shear nonlinearity: self-response at g=0.05 is 3.7% below the g=0.02 value (0.448 vs 0.465).

USER LEAD (2026-07-09): there is a DIP in R_blend at INTERMEDIATE angular separation, seen in emulator +
half-shear but likely NOT in constant(coherent). If the isotropic response dips at mid-sep where the
coherent doesn't, the isotropic average < coherent -> exactly the +0.039 gap. Testing via `audit_sep_dip.py`:
response vs FINE separation, isotropic (half-shear truth + emulator) vs coherent (constgold single-dominant
objects, r_sim - R_self(mag) vs nearest-nbr distance). Confirms the mechanism if isotropic dips & coherent doesn't.

Feasibility tests launched: within-sample transfer (fit deficit(R_blend) on cases 0-19 job 15005758 ->
apply to held-out 20-39 job 15005759, added --max-case) for an EARLY answer tonight; gold-standard is
apply rblend_corr_c0-39.npz to fresh 40-79 (job 15000850, ~overnight). Curve saved rblend_corr_c0-39.npz.

## 2026-07-09 (cont.3) — BLEND EMULATOR VINDICATED per-pair; deficit is R_flow or coherent ANISOTROPY

audit_blend_truth (job 15004788): emulator predict_on_pairs vs half-shear truth delta_et1/gamma(=0.2)
on 44.9M leakage-free gold pairs (cases 0-39). Per-pair the emulator MATCHES truth to <0.001 EVERYWHERE:
  r_s 13-24 pred/truth 0.1121/0.1116; 24-25 0.0244/0.0246; 25.5-26 0.0040/0.0042; 26-26.5 0.0018/0.0011
  (the old "-11% at r_s~26" is +0.0006 absolute -> negligible). By distance max dev -0.0017 at 2-3".
=> R_blend emulator is ACCURATE on our population. BRANCH B (emulator under-prediction) REJECTED.
  (The per-primary SUM table was CONFOUNDED: half-shear pairs each primary only with SECONDARIES, so
   the response catalogue has ~half the neighbours -> pred_sum 0.084 = 1/2 production R_blend 0.168,
   npair 8 vs ~16. Ignore its "truth-emu=-0.097 in q3"; the valid test is per-pair, where pred=truth.)

So the constgold q3 +0.047 deficit is NOT the emulator. Two branches remain, separated by the SELF audit:
  A) R_flow crowd-conditioning OVER-SUPPRESSES the self-response of blended objects.
  C) COHERENT ANISOTROPY: emulator+half-shear measure the ISOTROPIC (random-neighbour-direction) blend;
     constgold's coherent (all-angle-0) blend is larger. This is exactly the random-direction problem
     the constant sims were built to expose (cos(ghat,ghat')~0) -> structurally plausible, and the
     emulator has NO angle feature so it cannot represent it.
Decider = R_self_truth in q3 (self audit, job 15004817): with R_sim=0.4301, isotropic R_blend=0.219:
  R_self_truth~0.175(=R_flow) -> coherent_blend=R_sim-R_self=0.255 > 0.219 -> ANISOTROPY (branch C).
  R_self_truth~0.211          -> R_flow too low                          -> branch A.
Feasibility: A -> recalibrate flow crowd-conditioning (in-framework). C -> need an angle/coherence term
on R_blend (constant sims measure it) or conclude the isotropic-emulator framework can't hit 0.3% for
coherent shear. The cg_fit->apply-on-40-79 transfer test remains the agnostic feasibility arbiter.

## 2026-07-09 (cont.2) — TRUTH audit enabled: half-shear response catalogues cover the gold cases

Correction (user caught it): the constant sims have no response catalogue, but the HALF-SHEAR base
(lsst_sims_fs2_25876) DOES, and it shares the SAME fields/positions as constgold per case (case0:
identical 699,568 gals, RA 180.08918546...). So the decomposition truth EXISTS for the gold population:
  - response_catalogue_train.feather (cases 0-199): per-pair blend truth delta_et1  [gold 0-39 included]
  - self_response_catalogue_train_cases0_99.feather: per-object self-response truth  [gold 0-39 included]
  (numbered chunk files are cases 100-199 only -> my first map missed gold; the _train files have 0-39.)
The `ho` emulator is HELD-OUT on cases 40-199 (HELDOUT_MIN_CASE=40, gold 0-39 excluded) => cases 0-39
are leakage-free truth. Emulator target = delta_et1/gamma, gamma=values[1]-values[0]=0.2.

FRAME (Explore agent, response.py:287-305): delta_et1 = primary response projected onto the NEIGHBOUR's
shear direction; the regression uses NO angle feature -> emulator 'response' = isotropic <delta_et1/gamma>.
So the apples-to-apples coherent blend truth is Sum_neighbours delta_et1/gamma (delta_et2 ~0 by parity).
Per-object anisotropic coherent response is NOT reconstructable from the single random-direction half-shear
truth -> if emulator≈truth_sum but both < R_sim-R_self, the loss is coherent ANISOTROPY (branch C).

Launched `audit_blend_truth.py` (job 15004788): emulator predict_on_pairs vs truth delta_et1/gamma on
gold pairs (reg cuts rs[13,29] rp[18,28] Res[0,10] Rep[0.1,1.5] dist[0,10]), (A) per-pair by r_s/dist,
(B) per-primary SUM vs production R_blend by R_blend quantile. DECIDES: truth_sum-R_blend(emu) ~+0.047
in q3 => emulator under-predicts (branch B, recalibratable vs the half-shear truth directly); ~0 =>
emulator fine, q3 deficit is R_flow (branch A) or coherent anisotropy (branch C). Clean-bin r_s[18,24]
pred/truth~1 validates gamma. NEXT: mirror self_response audit (R_flow vs self-response truth).

## 2026-07-09 (cont.) — Multiplicity+dominance REJECTED w/ proper R_flow; deficit tracks R_blend; correction machinery built

cg_mfast (job 15002621, fast unbuffered replica) delivered the multiplicity/dominance table with
model-measured per-cell R_flow. q3 deficit is UNIFORM ~+11-12% across every split (n_pairs ≤17 +12.3%,
>17 +11.5%, 1-dominant +11.1%). Proper per-cell R_flow ABSORBS the dominance swing my offline
fixed-R_flow test showed (q1 1-dom R_flow=0.348 vs many-small 0.211) — confirming that swing was the
target-mag/R_flow confound, not a blend effect. Case-split replication: cases 0-4 q3=+9.1%, i.e. stable.

Absolute R-deficit (R_sim−R_flow−R_bl) by R_blend quantile: q1~0, q2~0.015, q3~0.047, q4~0(−0.008 on
cases0-10; +0.012 on full 40). Peaks at MODERATE R_blend => the error tracks blend STRENGTH, not
crowding/multiplicity/mag. An R_flow-suppression explanation would need a non-monotonic-in-crowding
error (wrong at q3's moderate crowd, right at q4's heavier crowd) => implausible; points to R_blend.

DEGENERACY: constant sims have NO response catalogue (only input/CrossMatch/Shapes/SExtractor), so the
per-cell R_flow/R_blend split cannot be broken without new (primary-only-sheared) renders. Reframed:
to hit 0.3% we don't need the A/B attribution — we need a correction that TRANSFERS out-of-sample.

Built fit/apply machinery in `validate_constant_with_blend.py` (edits confined there):
`--fit-rblend-corr OUT.npz` (per fine R_blend quantile bin: proper R_flow + deficit; writes curve;
fit-only early-return) and `--apply-rblend-corr IN.npz` (adds interp(deficit,R_blend) to R_blend term).
Launched FIT on cases 0-39 (job 15004337 `cg_fit`, 24 bins -> results/rblend_corr_c0-39.npz).
NEXT: when held-out 40-79 renders finish (job 15000850), build their blend/crowd/ood lookups and run
validate with --apply-rblend-corr rblend_corr_c0-39.npz => decisive out-of-sample m. Transfer to <0.3%
=> feasible; no transfer => residual is covariate-shift noise not captured by a blend-strength recal.

## 2026-07-09 — Residual localised to MODERATE blend; multiplicity REJECTED; robustness confirmed

Proper per-magnitude & per-R_blend-quantile R_flow table (job 14984541, `cg_dump`): the +1.67% is NOT
a global flow OOD problem. Isolated (blend-free) flow is well-calibrated (m_blend −0.7%). The whole
deficit sits in the moderate-blend R_blend quantile **q3: +9.1% ± 1.0%, carrying 73% of the global
deficit** (q1 +0.9, q2 +1.9, q4 +1.4). By distance closest-neighbour bin +8.4%.

Robustness (user Qs): (a) **not outlier-driven** (`q3_outlier_check.py`) — trimming top/bottom 1%
(34k obj) leaves q3 at +8.2%; top 0.001% carry −0.5% of the excess; median-case m_blend already +9.0%
(40 cases min −4% / med +9% / max +19.6%, std 6.2% → SE 0.98% ≈ bootstrap). (b) **not a few-case
fluke** — every case scatters ~symmetrically around +9%.

Multiplicity discriminator: built `blend_multiplicity_extnbrho_c0-39.feather`
(`build_blend_multiplicity.py`, job 15000757: n_pairs, rb_max, rb_top2 per obj, 22.5M pairs).
Offline preliminary (`mult_prelim.py`) **REJECTS multiplicity/super-additivity**:
- R_blend quantiles do NOT sort by neighbour count — all q have ~16 neighbours (median n_pairs≈17).
  What rises is **dominance**: frac(one nbr ≥70% of |R_blend|) 47%→52%→**66%(q3)**→78%(q4). "Moderate
  blend" = one moderately-strong close/bright neighbour, not many-moderate.
- Within q3, deficit FALLS with n_pairs (n≤17 +11.1% vs n>17 +7.0%) — opposite to super-additivity,
  consistent with the toy linearity proof. Deficit concentrates in single-dominant-neighbour objects
  (but that split is R_flow-confounded via target mag; needs proper per-cell R_flow).

Mechanism now narrowed to two, to be separated by the proper per-cell R_flow in `cg_mult`
(job 15000779, running, ~3h, grep-buffered): crowd-flux **R_flow over-suppression** at moderate
crowding vs emulator under-predicting the **strong-single-neighbour R_blend**.

Held-out cross-validation launched (user suggestion): `job_fs2_constant_xval.sh` (job 15000850)
generates constant cases **40–79** (seeds 163–202, new independent fields, same ±0.02 config, no
clobber of 0–39, ~14h). Will rebuild blend+mult lookups on 40–79 and re-validate → true out-of-sample
confirmation of the effect and any derived correction. Added `--mult-lookup` + multiplicity table +
case-half replication table to `validate_constant_with_blend.py` (all edits confined to that file).

## 2026-07-08 — Marginal debiasing FAILS; residual is emulator COVARIATE SHIFT, not physics

Ran the full solution pipeline. Verdict: the emulator self-debiasing does NOT close the constgold m.
- `validate_constant_with_blend.py` RAW R_blend (job 14973974): m = **+1.69% ± 0.27%** (anchors baseline).
- CORRECTED R_blend, Delta(r_s,distance) from the emulator's own held-out validation (job 14973975):
  m = **+1.82% ± 0.27%** — statistically unchanged, if anything slightly WORSE. Corrected mean R_blend
  moved only 0.3105 -> 0.3114 (+0.29%): the validation biases (r_s~26 −11%, faint r_s +70..+700%) nearly
  cancel in the net, so the marginal correction is tiny AND mis-signed for constgold.

Binned constgold breakdown (CORR run) localises the residual:
- FLOW is clean: truly-isolated (R_blend~0) m_bare ≈ −1.4% ≈ 0 -> self-response is NOT the cause.
- Residual is in the BLEND term, concentrated in close pairs (distance tercile d1: **+8.1%**, growing
  monotonically d3 −1.2% -> d2 +4.0% -> d1 +8.1%) and moderate-high R_blend (q3: **+9.6%**, ~2/3 of the
  global deficit) — while EXTREME blend (q4) is **+0.0%** (perfect). Non-monotonic (q3 bad, q4 good) =>
  not clean physics super-additivity.

DECISIVE toy (`toy_close_pair_scan.py`, job 14980195): pushed the emulator-free decomposition to
sub-arcsec separations + faint-crowding, the untested gap (prior sweeps stopped at 1.2"):
- EXCESS ≈ 0 EVERYWHERE: equal-flux pair 0.4"→2.0" all <1.2σ; faint nbr, faint target (q3 analog) ≤2%;
  multiplicity ladder to 8 neighbours ≈ 0%. **Super-additivity is REJECTED even at 0.4".** The linear
  sum R_full = R_self + Σ R_blend is EXACT.

=> By elimination the entire +1.7% is emulator per-pair R_blend accuracy, and the correction's
non-transfer (built on validation, worsens constgold) is the signature of COVARIATE SHIFT: at fixed
(r_s,distance) the residual depends on other covariates (target mag r_p, size), and the constgold pair
population differs from training along them. Running `emu_domain_shift.py` (job 14981379) to prove it:
(A) held-out residual 2D by (r_s x r_p), (B) train-vs-constgold pair distributions, (C) bias re-weighted
by each population. If confirmed, sub-percent is feasible only by a MULTI-covariate correction or an
emulator retrain matched to the constgold pair domain — NOT by the marginal (r_s,distance) route.

## 2026-07-08 — CONFIRMED: R_blend emulator has a faint-neighbour bias (the constgold cause)

`scripts/confirm_emulator_bias.py` (job 14973798): reproduced the regression emulator's own held-out
test set (44.9M pairs, extnbr_ho, target delta_et1/shear) and compared predicted vs true MEANS:
- GLOBAL mean bias = +0.65% (NOT a uniform offset).
- By NEIGHBOUR magnitude r_input_s: r_s<25.7 = -0.0% (clean); r_s~26 = **-11.1%** (under-predict);
  r_s~27 = +71%; r_s~28-29 = +737% (relative blowups on ~0 true response).
- By distance: fine <7", +300% at 7-8.7" (again near-zero true response).
=> The user's remembered "~10% off" is real and localised to the FAINT-neighbour end (r_s~26 under by
11%), NOT the bright end (r_s<25.7 clean). The global cancels, but the per-neighbour-mag structure means
the SUMMED R_blend is biased on the constgold neighbour population. r_s~26 neighbours are abundant (near
the detection limit); their -11% under-prediction -> summed R_blend too low -> linear R_flow+R_blend
under-shoots -> POSITIVE constgold m. Sign and rough size match the +1.71%, and it is fully consistent
with linearity holding: the linear sum is fine, the R_blend INPUT is faint-neighbour-biased.
Solving next: debias the emulator with its own validation (additive Delta by r_s x distance), rebuild
R_blend on the constant set, re-run linear constgold -> expect m -> sub-percent.

## 2026-07-08 — CORRECTION: the coherent response is LINEAR; super-additivity RETRACTED

User challenged the super-additivity story (dilution should make crowding REDUCE per-pair response,
not create a collective boost). Re-ran the CLEAN emulator-free decomposition `toy_blend_decompose.py`
(shear target-alone, each-neighbour-alone, all-together; excess = R_full - (R_self + Sum R_blend_j)) at
REALISTIC separations (1.2") and S/N, sweeping flux 500/800/2000 and 1/2/4/mixed neighbours
(`logs/toy_decomp_sweep.out`):
- EXCESS is consistent with ZERO everywhere: e.g. 4-nbr bright +0.000+/-0.003, 4-nbr faint -0.001,
  2-nbr faint -0.000. Tightest bins are 0.0-0.3 sigma. Only outlier: very-faint(500)+mixed-dist
  +11% at 1.3 sigma -- NOT significant.
- => The coherent response IS LINEAR: R_full = R_self + Sum_j R_blend_j to within noise. There is NO
  super-additivity at realistic parameters. (Consistent with the moment argument: total scene
  second-moments transform coherently, Q(g)=M Q(0) M^T, which PREDICTS a linear response.)

RETRACTIONS (earlier 2026-07-08 entries were WRONG on the mechanism):
- The "+56%..+569% super-additive excess" from `toy_blend_linearity.py` was an ARTIFACT: noiseless
  extreme close-pair (1.0") AND a pathological marginal definition (marginal_j = R_pair_j - R_self,
  so R_linear subtracts R_self N times). Not physics.
- "Linear R_flow+R_blend is structurally infeasible for sub-percent" is WRONG. Linear is fine; the
  inputs were miscalibrated.
- "A nonlinear combiner g(R_self,R_blend,mag) is REQUIRED" -- the nonlinearity was COMPENSATING for a
  miscalibrated input: the emulator R_self is off by ~2x (0.15 vs true 0.066 in the crowded bin; 0.42
  vs ~1.0 for bright). A linear model can't undo a magnitude-dependent 2x input error, so it "needed"
  magnitude/nonlinearity -- fixing the INPUT, not capturing nonlinear physics. With correct R_self and
  R_blend, linear suffices.

CORRECTED DIAGNOSIS of constant-gold +1.71%: input miscalibration, NOT the linear combination.
R_flow ~= R_self_true (flow is ~right, even crowded: q4 flow 0.067 vs measured 0.066), so the shortfall
is R_blend (emulator supplies ~0.17 vs true ~0.19, ~11% low; m*R_total ~ 0.008 global under-supply).
Prime suspect: the emulator's magnitude-domain cut DROPS faint/OOD neighbours that do contribute
(`toy_faint_neighbour`: a 0.25-flux neighbour adds R_blend~0.06; constgold ood_flux binning already
shows those neighbours move m). Dilution itself does not bias the global mean (it is in the emulator's
training average).

ACTIONABLE (vindicates the user's staged LINEAR plan): stage-1 self-response from the flow on
half-shear (already ~right); stage-2 fix the emulator R_blend under-supply (faint/OOD-neighbour
coverage); sum LINEARLY. The scene-level / nonlinear-combiner work is not needed for the mechanism,
though the field-based scene features remain a valid way to CALIBRATE R_blend's crowding dependence.
Caveat: toy uses round equal galaxies; real diversity untested, but the moment argument holds generally.

## 2026-07-08 — Toward an ANALYTICAL / DERIVED combiner (user: prefer derived from moments)

Goal: replace the black-box XGBoost combiner with a closed-form/derived model. Status of attempts:
- Fitted polynomials in (R_self_emu, R_blend, magnitude), lstsq CV-by-case (`analytical_combiner.py`,
  job 14970711): linear WORST 41%; adding mag cross-terms (P4/P5, 11-15 params) only reaches WORST ~10%.
  A rational/Pade form (mis-specified) failed (WORST 78%).
- Fitted physical MIXTURE R_full=Rself(mag,size)*(1-w(rb))+Rcap*w(rb) (`analytical_v2.py`, 14972403):
  WORST 31% -- the magnitude-sigmoid Rself is too crude; the real self-response driver is SIZE (PSF
  resolution), not magnitude.
- KEY realisation from the (mag x r_blend) EXPLORE table: the emulator R_self is badly miscalibrated to
  the true isolated response (bright ISO R_full~1.0 vs R_self_emu~0.42; faint ~0.15 vs ~0.11), which is
  why R_self-based fits cap out. And isolated R_full tracks SIZE via PSF dilution (bright/big ->~1,
  faint/small ->~0.15).

DERIVED model (physics, not fitted curve): the total scene second-moments transform under coherent
shear exactly like a single object (Q(g)=M Q(0) M^T), so the measured response should be the standard
PSF-resolution factor of the SCENE moments:  R = R0 * [T_scene/(T_scene+T_psf)] * (1 - kappa*e_scene^2),
with T_scene = flux-weighted Sum_k f_k (2 sigma_k^2 + |d_k|^2) / Sum_k f_k (target + neighbours, incl.
separations) -- all g=0-computable. Isolated response ∝ size/(size+PSF) (the magnitude dependence via
size-mag relation); crowding adds neighbour f*(sigma^2+d^2) to T_scene -> dip (dilution) then rise
(super-additive) fall out with only physical constants (R0, T_psf, measurement-weight scale).
Implemented `scripts/scene_moments.py` (KDTree over input fields, weighted moment traces + quadrupole
at scales 0.4/0.6/0.9/1.3", r_max=5", cached to `results/scene_moments.feather`; job 14972707) and
`scripts/derived_fit.py` (fits only R0,T_psf,kappa per scale, CV by case). If it reaches sub-percent
with physically-sensible T_psf -> a derived closed-form combiner; else iterate/conclude.

## 2026-07-08 — Minimal combiner found: g(R_self, R_blend, magnitude) reaches sub-percent (loop)

Faithful stage-2 test on the ACTUAL emulator outputs (`scripts/build_self_lookup.py` ->
`results/self_lookup_const_c0-39.feather` with the production tag `lsst_r_extnbr_ho`; combiner in
`scripts/combiner_self.py`, job 14967861/14967912). Predict measured coherent R_full from the emulator
self-response R_self and blend-response R_blend, 5-fold by case, per-r_blend-bin AND per-magnitude m:
- PROD 1:1 sum R_self+R_blend: global +25% (emulator R_self 0.216 + R_blend 0.153 = 0.369 << R_full
  0.461; note this uses the EMULATOR self-response, not the SBSI flow's R_flow~0.29, so it is not the
  production +1.71% -- the scale is just wrong, which a fitted combiner absorbs).
- LINEAR a*R_self+b*R_blend+c: per-r_blend -18%..+13%, per-mag +41%..-27% -> infeasible.
- g(R_self,R_blend) nonlinear (XGB d3): FLAT in r_blend (+/-0.5%) but +35%..-28% by MAGNITUDE --
  the two response scalars do NOT encode magnitude, so a 2-input combiner is insufficient.
- g(R_self,R_blend,magnitude) nonlinear (XGB d3): FLAT sub-percent on BOTH axes -- r_blend +/-0.44%,
  magnitude +/-0.4%, held out by case. THIS is the minimal sufficient combiner.
- Adding Re (d4) doesn't improve it.

LOOP CONCLUSION (measurement coherent response): sub-percent IS achievable under the current framework,
but NOT by linear R_flow+R_blend addition -- it requires a small nonlinear recalibration
g(R_self, R_blend, magnitude) of the two existing emulator outputs (interpretable, ~3 inputs, depth-3,
cross-validated by case). Linear is structurally capped because R_full is non-monotonic in crowding
(dilution then super-additivity). Headline (job 14967912): held-out m = +0.008% +/- 0.255% (per-case SE, 40 cases; spread std 1.61%);
additive c = -0.00018 global, all r_blend bins within +/-7e-4. => SUB-PERCENT ACHIEVED for the coherent
MEASUREMENT response. Preds saved: `results/combiner_self_heldout.npz`.
Remaining before production: (1) end-to-end constgold m/c with selection using g(.) (my metric is
<R_true>/<R_pred> on mutual-detection objects); (2) the SELECTION-response term (separate classifier);
(3) independence -- trained/validated on the only 40 constant cases; more constant realizations needed.

## 2026-07-08 — Linear vs nonlinear combiner: linear infeasible for sub-percent (loop)

Testing whether the user's proposed stage-2 (self-response + R_blend combined LINEARLY) can reach
sub-percent, on cached constant features (`scripts/linear_vs_nonlinear.py`, `scripts/combiner_form.py`,
5-fold-by-case held-out; jobs 14967805/14967819):
- LINEAR (OLS) own+r_blend, or +scene, or +explicit rb^2/rb*own interaction terms: per-r_blend-bin m
  stays -11%..+7% (worst in the MID bins). A fitted linear combiner CANNOT flatten it.
- Reason: R_full is NON-MONOTONIC in crowding -- <R_full> by r_blend bin = 0.446(ISO), 0.314, 0.304,
  0.383, 0.752(>=.25). It DIPS at moderate crowding (dilution ~1/scene-trace, the toy_dilution result)
  then SPIKES at high crowding (super-additivity). A linear model can't fit that U-shape; the
  production's linear R_flow+R_blend structurally can't hit sub-percent.
- Minimal nonlinear form that DOES work: depth-3 XGBoost on just [mag, Re, r_blend] -> per-bin m
  ~+/-1% (r_blend) and ~+/-0.5% (magnitude), held-out by case. A 2D LUT g(mag8 x rblend8)=64 cells is
  flat in r_blend (tautological, r_blend is a LUT axis) but +/-2% by magnitude (too coarse). Full
  XGBoost (all feats, depth7) reaches +/-0.2-0.35%.
- CONCLUSION so far: sub-percent IS achievable but REQUIRES a nonlinear combiner; a low-parameter
  3-input model (magnitude, size, r_blend) already gets ~+-1% and generalizes (depth-3, held-out cases).
  Linear addition is the ceiling. Next: build the emulator's real self-response R_self on the constant
  cases (`build_self_lookup.py`, job 14967841) and test the clean, directly-actionable 2-input combiner
  g(R_self, R_blend) linear vs nonlinear -- the faithful production form.

## 2026-07-08 — Scene-level coherent-response forward model (lever-2 prototype)

Production R_blend already sums the per-pair emulator over ALL aperture neighbours
(`build_blend_lookup.py`: `predict_response(t,t)` then `groupby.sum()`), so the under-count is NOT
"too few neighbours" -- it is the aperture/domain cut (drops faint/OOD neighbours) plus the
super-additivity a per-pair SUM cannot represent (toy_blend_linearity). The principled fix is to
model the coherent response R_full DIRECTLY (the field shear is coherent in a real survey, so R_full
is the physically relevant response), as a forward model of own properties + aggregate g=0 scene
features, cross-validated across independent case realizations.

New: `scripts/scene_coherent_model.py` + `jobs/job_scene_coherent.sh`.
- Data: `constant_response_catalogue_train.feather` (cases 0-39; one row per (target,neighbour) pair,
  per-object coherent `response`). Aggregate to per-object: target R_full; own feats
  (r,Re,sersic_n,q,z); SCENE feats (n_nbr, sum neighbour flux, sum flux/d^2, sum flux/d, flux_ratio,
  min/mean neighbour distance, light-weighted scene_trace, own flux).
- k-fold BY CASE (train 32 cases, predict held-out 8) -> fully held-out R_pred for every object.
  Sanctioned cross-case validation (SBI_shear_response.md), directly comparable to constgold cases 0-39.
- Metric: held-out m = <R_true>/<R_pred>-1 globally + per r_blend bin (same axis as constgold); c from
  sum_et. Ablations: global-mean, OWN-only (~self), OWN+SCENE (coherent forward model). If OWN+SCENE
  is flat per r_blend bin -> scene features capture the coherent/super-additive response and a single
  forward model REPLACES R_flow + R_blend; if the crowded bin stays biased -> g=0 scene features are
  insufficient (needs the render), also informative.
- Smoke test (800k rows = only cases 0-2, 1 case/fold, undertrained) ran clean but is not
  interpretable; full 40-case run submitted as `14964592` (n_est=600, depth=7, 5-fold).

RESULT v1 (14964592): NEGATIVE, but diagnostic. own+scene barely beat own-only; both predict a flat
~0.45 while true R_full swings 0.30->0.75 across r_blend (held-out per-bin m: ISO -12%, mid -24%,
q3(=[0.10,0.25)) -10%, >=0.25 +67%). Root cause: the constant response catalogue records at most ONE
neighbour per object (`max n_nbr=1`; 75% have 1, 25% isolated), so its "scene" features carry NO
multi-neighbour crowding. Global m is trivially ~0 (XGBoost squared-error mean-unbiased) -- the
per-r_blend-bin m is the only signal and it says the features were impoverished, not that the idea fails.

v2 -- field-based scene features. New `scripts/scene_coherent_field.py` + `jobs/job_scene_coherent_field.sh`:
rebuild scene features from each case's full input `gals_info` field (~700k galaxies) via a cKDTree over
ALL neighbours within r_max=10" (INCLUDING faint ones below the emulator's r<28 cut). Per target: own
morphology (r,Re,n,q) + n_nbr, sum neighbour flux, sum flux/d^2, sum flux/d, sum flux*Re^2, nearest
distance/flux, flux_ratio, light-weighted scene_trace, and per-shell (2/4/7/10") counts+fluxes.
Smoke test (3 cases, undertrained): <n_nbr> jumps 0.75 -> 16.75 and per-bin m improves sharply --
ISO -6.5%, [0.02,0.05) -3%, [0.05,0.10) -10%, [0.10,0.25) -13%, >=0.25 +24% (was +67% in v1). So g=0
multi-neighbour scene info DOES predict much of the coherent response; the crowded bin still
under-predicts (0.60 vs 0.74, the super-additivity signature) but far less. Full 40-case run: `14964859`.

RESULT v2 full (14964859): held-out per-r_blend-bin m = ISO -5.5%, [0.02,0.05) -5.5%, [0.05,0.10)
-11.1%, [0.10,0.25) -7.5%, >=0.25 +18.5% (R_true 0.752 vs R_pred 0.635). Global trivially 0 (fit to
R_full). So field scene features capture most of the coherent response but the direct XGBoost
UNDER-predicts the extreme-crowded tail by ~18% (squared-error regression-to-mean on rare high-R
objects). Note the existing emulator handles that extreme tail BETTER (constgold top bin +1.1%), so a
direct scene model does not beat the R_flow+R_blend decomposition in the tail. Next: STACK -- add the
emulator's own r_blend prediction as a feature (keeps its physics tail, learns the super-additive
correction on top) + a tail-weighted variant. Script now caches the (expensive) field-feature table to
`results/scene_field_features.feather` so XGBoost variants iterate cheaply. Run: `14964957`
(A scene-only, B scene+r_blend stack, C stack+tail-weighted).

RESULT (14964957) -- STACK WINS. Adding the emulator's own r_blend prediction as ONE feature on top of
the field scene features (variant B) gives held-out per-r_blend-bin m FLAT and sub-percent:
ISO -0.05%, [0.02,0.05) -0.11%, [0.05,0.10) -0.35%, [0.10,0.25) +0.14%, >=0.25 +0.12% (vs the
R_flow+R_blend decomposition's +1.71% global / q3 +8.9%, and vs scene-only A's -11%..+18.5%). The scene
features supply the super-additive correction the emulator misses; the emulator's r_blend supplies the
per-pair physics tail; together they are flat. (Variant C tail-weighting mis-scaled -> diverged; ignore.)

Non-tautological validation (r_blend is now both a feature and the bin axis, so its own bin-flatness is
partly favoured -- checked INDEPENDENT axes, cached features + saved preds):
- by own r-mag [18..28]: m = -0.02%, +0.04%, +0.01%, -0.23%, +0.13% (flat sub-percent).
- by n_nbr: flat where populated (10-20 nbr -0.01%, 20-40 +0.12%).
- per HELD-OUT case (genuine cross-realization generalization): m mean -0.004%, std 1.6%, range
  [-3.6%,+3.4%] over 40 cases -> unbiased, ~0.25% on the 40-case mean.
- additive c = <sum_et>/2 = -0.00018 global, few*1e-4 per bin -> no additive bias introduced.

Conclusion: the COHERENT response R_full is predicted, held-out and cross-case, to sub-percent flatness
across crowding AND magnitude by a single forward model of [emulator r_blend + g=0 multi-neighbour scene
features]. This is the principled lever-2 fix at the forward-model level -- it fixes the crowded-bin
under-supply that neither the flow nor a scene-only model could. Remaining to productionize/verify:
(1) run the REAL constgold pipeline (validate_constant_with_blend.py) with this stacked model as the
response for an end-to-end m/c under the actual selection (my metric is <R_true>/<R_pred> on
mutual-detection objects, not yet the full selected-sample estimator); (2) an independent constant-case
set for a non-CV test (only cases 0-39 constant renders exist today); (3) fold the selection-response
term in. Artifacts: `scripts/scene_coherent_field.py`, `jobs/job_scene_coherent_field.sh`,
`results/scene_field_features.feather`, `results/scene_coherent_field_heldout.npz`.

ABLATION (14965127/14965231, `scripts/scene_ablation.py` on cached features) -- the decisive read:
- own-only: ISO -14%, >=.25 +83% (self-response only; misses all blend).
- own+scene (NO emulator): ISO -5.6%, >=.25 +19% -- raw g=0 scene features do NOT flatten the tail.
- own+r_blend (NO scene features): FLAT -- ISO +0.04% .. >=.25 -0.02%; and on INDEPENDENT axes flat too
  (r-mag -0.09%..+0.02%; per-held-out-case mean -0.02% std 1.6%). Only the sparse n_nbr 0-10 bin lags (-3.9%).
- own+scene+r_blend (B): identical to own+r_blend except it fixes the sparse tail (n_nbr 0-10: -3.9%->-1.5%).
- B feature importance: scene 61%, own 30%, r_blend 9% -- scene feats are USED but largely redundant with
  (own,r_blend) for the binned m; they only matter in the sparse-neighbour regime.
- 2-fold (train20/test20) B stays flat -> not data-hungry.

LEVER-2 CONCLUSION. The coherent response R_full is predicted FLAT to sub-percent (across crowding,
magnitude, and held-out cases) by a NONLINEAR learned model of [self-response predictors (own morphology
~ R_flow) + the emulator's r_blend]. The production's +1.71% / q3 +8.9% failure is the LINEAR addition
`R_total = R_flow + R_blend`, which misses the super-additive self<->blend coupling the toys identified.
The emulator's r_blend is already a SUFFICIENT crowding statistic -- it under-supplies in absolute
(per-pair-sum) terms, but a nonlinear recalibration `g(R_flow, R_blend, own)` recovers the true coherent
response. So the minimal principled fix is to REPLACE the linear sum with a small learned nonlinear
combiner g(.), cross-validated across cases (not empirical m-removal). Rich multi-neighbour scene
features are NOT needed except a marginal sparse-neighbour gain.
Caveats before productionizing: (1) statistical floor ~0.25% with only 40 constant cases (per-case std
1.6%); (2) metric here is <R_true>/<R_pred> on mutual-detection objects -- must run the real
`validate_constant_with_blend.py` end-to-end with g(.) as the response, and fold in the SELECTION-response
term separately; (3) ideally an independent constant-case set (only 0-39 exist) and a depth-shifted
crowding test for population robustness. Next concrete step: build per-object R_flow on the constant cases
and fit/validate `g(R_flow, R_blend, own)` in the actual constgold pipeline.
Artifacts add: `scripts/scene_ablation.py`, `jobs/job_scene_ablation.sh`.

## 2026-07-08 — Re-confirmed lever-2 (coherent blend) toy validations; numbers logged

The coherent-blend under-supply (lever 2) was already investigated with a suite of GalSim
postage-stamp toys in `scripts/toy_*.py` (never Slurm-logged; SBSI is not a git repo). Re-ran two
to capture the actual numbers, which pin the mechanism:

- `toy_faint_neighbour.py` (shear ONLY a probe neighbour @1.2", target flux 1200): probe R_blend by
  flux ratio = 1.0->0.577, 0.5->0.349, 0.25->0.062, 0.10->-0.005, 0.04->-0.034. So moderately-faint
  neighbours (0.25-0.5 flux, 0.75-1.5 mag fainter) DO add real positive R_blend; the emulator's r<28
  domain cut drops some of them -> a linear under-count of the right (positive-m) sign.
- `toy_blend_linearity.py` (target + N neighbours, coherent +/-g, ngmix on the target; noiseless,
  seeded): the emulator's linear per-pair superposition FAILS in crowded scenes. For 4 equal nbrs on
  a 1.0" ring: R_self=+2.60, sum(per-pair marginals)=-5.13, R_linear=-2.52, but R_full(coherent)=+0.98
  -> EXCESS=+3.50 (+358% of R_full, super-additive). 6 nbrs @0.8-1.5": excess +569%. 2 bright nbrs on
  the g1 axis: excess +56%. (Noiseless close-blend magnitudes are exaggerated; the calibrated size in
  the real sim is the measured ~0.019 / ~11% global coherent under-supply, concentrated in crowded
  R_blend bins.)

Conclusion: lever 2's under-supply is real and validated on toy sims, and its dominant cause is
COLLECTIVE SUPER-ADDITIVITY of the coherent blend response, which the current per-pair-summed
`BlendingPredictor.predict_response` architecturally cannot produce (plus a smaller faint/OOD-neighbour
linear under-count). So the lever-2 fix is NOT "sum more per-pair terms" — it needs a crowding-aware
(collective) coherent-response term. Related in-sim diagnostic already in production: constant-gold bins
by `ood_flux_bright/faint` (the dropped-neighbour channel). - `toy_dilution_scaling.py` (shear ONLY a probe @1.2", add M equal neighbours, flux 1000/sky 10,
  S/N~22): R_blend_probe * sigma^2 (sigma^2 = scene adaptive-moment trace/2 from the noiseless g=0
  blend) = 0.282 (M=0), 0.236 (1), 0.214 (2), 0.206 (4), 0.201 (6). So R_blend ~ C/sigma^2 holds to
  ~5% in the CROWDED limit (M>=2, C~0.20) but C rises ~40% toward the isolated pair (0.28). A per-pair
  1/scene-trace dilution is thus a decent crowded-regime closed form (computable from g=0
  fluxes+sizes+positions) but not exact, and -- crucially -- it corrects each per-pair MARGINAL, so it
  cannot supply the COLLECTIVE super-additive term from toy_blend_linearity. Net: a closed-form g=0
  crowding correction is PARTIALLY available (per-pair dilution) but does not close the coherent gap
  on its own; the dominant super-additive collective piece needs a scene-level (all-neighbour)
  coherent-response model.
Companion toys not re-run this session: `toy_blend_decompose.py`, `toy_sersic_superpose.py`
(profile-mismatch), `toy_model_calib.py` (flow R_flow vs toy R_self isolated agreement).

## 2026-07-07 — Relative-error (multiplicative-m) response loss

Diagnosis (root cause of the residual constant-gold m and the g=0.02 half-shear tilt):
- The response loss minimized ABSOLUTE per-bin error `(R_model - R_sim)^2 * cnt_b`
  (`epoch_response`, formerly line 385). But the quantity we actually care about is the
  RELATIVE (multiplicative) bias `m = R_sim/R_model - 1`.
- Absolute error is dominated by the high-response isolated/bright cells and tolerates large
  RELATIVE errors in the small-response crowded/faint tail. Consequence: absolute errors cancel
  in the count-weighted GLOBAL response for the TRAINING population (half-shear g=0.05 global
  `m=+0.02%`), but leave a crowding TILT (g=0.05 SNC bins: ISO `-1.1%`, q2 `+3.6%`, q3 `+5.5%`).
- Any population reweighting then breaks the cancellation: constant-gold is heavier in the
  crowded bins, so the tilt does not cancel -> `m=+1.71%` (lam300) dominated by R_blend q3
  (`m~+9%`). The g=0.02 half-shear amplifies the same tilt (`+3.1%`, q2 `+12%`).
- Confirmation the flow CAN represent the crowded response: raising lam 300->1000 fixes q3
  (`-0.8%`) but over-pulls ISO (`+1.1%`, constant-gold ISO `+9.7%`) -> a single global lambda
  on an absolute loss cannot calibrate isolated and crowded simultaneously; a cancellation, not
  a flat calibration.

Fix (principled + minimal; no empirical m-subtraction):
- `scripts/train_measurement_model.py`: added `--response-error {absolute,relative}` and
  `--response-rel-floor`. Relative penalizes `((R_model-R_sim)/max(|R_sim|,floor))^2 * cnt_b`,
  i.e. per-bin `m` directly. Its gradient scales as `1/R_sim^2`, so it up-weights the
  small-response crowded/faint tail automatically and drives `m->0` uniformly per bin, robust to
  population reweighting. One-line loss change; reuses the existing SNC target grid. Choice
  recorded in `model_config["response_error"]`.
- `sbs_shear/measurement_model.py`: `build_flow` now also drops the training-only
  `response_error` config key on load.
- Added `jobs/job_train_crowd_snc_relerr.sh` (relative loss, central delta=0.02, SNC target).

Runs (queued; training ~2h then validations):
- lam=15: train `14958943` -> constgold `14958944`, half-shear g=0.05 `14958945`, g=0.02 `14958946`.
- lam=60: train `14958947` -> constgold `14958948`, half-shear g=0.05 `14958949`, g=0.02 `14958950`.
- lambda rescaled from the absolute-loss sweet spot (~300) by the ~1/R_sim^2 gradient factor;
  lam=15 keeps the crowded-bin pull near the old lam~300 while relaxing isolated, lam=60 is the
  stronger-enforcement hedge. Success criterion: FLAT per-bin half-shear m AND constant-gold
  |m|<~0.3% without the ISO/q3 cancellation.

RESULT: NEGATIVE. The relative-per-cell loss REGRESSED constant-gold.
- constant-gold m: lam15 `+2.22% +/- 0.27%`, lam60 `+4.09% +/- 0.28%` (vs absolute lam300 `+1.71%`).
  Crowded R_flow DROPPED (q3 `0.176 -> 0.170 (lam15) -> 0.156 (lam60)`), the OPPOSITE of intended.
- half-shear g=0.05 tilt got slightly worse (lam15 q3 `+7.2%` vs abs `+5.5%`); g=0.02 GLOBAL improved
  (lam15 `-0.23%` vs abs `+3.11%`) but that is redistribution, not a real fix.
- Root cause (confirmed by weighting the 90-cell SNC target grid): 24 of 90 cells have `|R|<0.05`
  and 25 have `R<0` (noise-dominated low-count crowded/faint cells). The relative weight
  `cnt/max(|R|,0.05)^2` puts **84% of its mass on the |R|<0.05 cells** (abs loss: 32%) and 40% on
  `R<0` cells; the effective target it pulls toward is `<R>=0.012` (true population `<R>=0.282`).
  So the floor + per-cell division turned the loss into a noise-chaser that drags the response to
  ~0. Dividing by noisy small-magnitude per-cell targets is the failure mode.
- Reframing from the numbers (model-independent global decomposition): measured self-response
  (half-shear g=0.05) `R_self=0.2816`, BlendEMU `R_bl=0.1694`, coherent total `R_coh=0.4698`.
  Coherent excess BlendEMU SHOULD supply = `R_coh - R_self = 0.1882`, actual `0.1694` ->
  **BlendEMU under-supplies the coherent blend response by ~0.019 (~11%), concentrated in crowded
  bins**. The flow's self-response is globally excellent (g=0.05 `+0.02%`) but per-bin tilted
  (q3 self `+5.5%`). So the constant-gold residual is a SUM of a flow crowded-under-response AND a
  BlendEMU coherent under-supply; the flow is near its useful ceiling (even a perfectly flat flow
  leaves a coherent-gap floor of order `+0.5..0.8%`).
- constgold R_flow is central `(mp-mm)/2g` (validate_constant_with_blend.py:187-189), so the
  cross-harness R_flow gap (`0.2925` constgold vs `0.2709` half-shear at g=0.02) is population
  (gold constant-render vs g=0.02 test set), not an operator bug.
- Cancelled `14958950` (lam60 g=0.02 half-shear) as a confirmed regression with no info value.
- Decision: STOP cheap loss-metric reweighting (dead end). Two real levers remain, one crossing the
  SBSI/blendemu scope line, so surface to the user rather than spend more GPU:
  (1) per-object low-noise SNC self-response regression (in-scope; supervises each object by its own
      antithetic low-variance target, no binning and no dividing by noisy cell means -> the correct
      way to flatten the crowded self-response); ceiling still ~+0.7% from the coherent gap.
  (2) close the BlendEMU coherent R_blend under-supply in crowded bins (higher leverage; the
      `blend_lookup_*` products are built in SBSI from blendemu catalogues, so partly in-scope).

## 2026-07-07 — SNC/central response target diagnosis and retrains

Diagnosis:
- The full-200 retrain regression was not simply "more cases are worse." The full raw response target
  made the weighted `r_blend` q3 self-response target lower (`0.1888 -> 0.1856`), while the SNC target
  raises it to `0.1993`. The previous full target was therefore a more precise version of the wrong
  one-sided estimator for the current validation operator.
- The training loss had also used a forward `0 -> +0.05` response difference, while constant-gold and
  the trusted half-shear checks use a central `(+g - -g)/(2g)` operator at `g=0.02`.
- Selected-population `r_blend` distributions are mostly aligned between g0 train and g=0.02 half-shear.
  Constant-gold with the extended-neighbour lookup has the same mean `R_blend` (`0.1694` vs `0.1684`)
  but is slightly heavier in the `0.063-0.25` training edge bin, which amplifies the q3 residual.
- Constant-gold fixed-edge accounting shows the required term `R_sim_const - R_blend_emulator` is above
  the SNC half-shear self-response target in every `r_blend` bin; for `0.063-0.25`, required is about
  `0.226` versus SNC target `0.199`. So the remaining q3 residual is partly flow under-response and
  partly additive BlendEMU/coherent-population mismatch.

Code changes:
- `scripts/compute_response_target_blend.py`: added optional `--snc-lookup`, `--snc-cols`, and
  `--max-case`; SNC targets use `[e(g)-e(0)].ghat/g`.
- `scripts/train_measurement_model.py`: added `--response-difference {forward,central}` and central
  shifted contexts; checkpoints store the response-difference metadata.
- `sbs_shear/measurement_model.py`: checkpoint loader now ignores training-only `response_difference`;
  added fallback feature set `g0_crowd_flux_rblend` (`nbr_flux_near`, `nbr_flux_far`, and `r_blend`).
- `scripts/validate_constant_with_blend.py`: added `--rblend-edges-npz` fixed-edge diagnostics.
- Added jobs: `job_resp_target_crowd_snc.sh`, `job_train_crowd_snc_central.sh`,
  `job_validate_crowd_snc.sh`; updated `job_constgold_fulltrain.sh` to print fixed-edge bins.

Runs:
- `14948167` completed: wrote `results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz`
  with SNC match `99.31%` and global `R=0.2816`.
- `14948314` completed on an epoch-10 lam300 snapshot:
  constant-gold improved from full200 `m=+2.13% +/-0.27%` to `m=+1.05% +/-0.27%`.
  The residual is still dominated by `R_blend` q3 (`m=+7.4%`).
- `14948168` running: lam300, SNC target, central `delta=0.02`, full g0 train. Dependent validations:
  `14948170` (`g=0.05` SNC), `14948171` (`g=0.02` SNC), `14948172` (constant-gold).
- `14948546` running: lam1000 hedge with the same target/operator. Dependent validations:
  `14948547`, `14948549`, `14948548`.
- `14948576` pending: constant-gold validation of a mid-run lam300 snapshot
  `models/measurement_flow_g0_ngmix_crowdflux_snc100_central02_mid_lam300_v1.pt`.

## 2026-07-06 — Full 200-case crowd-flux flow retrain launched

Added Slurm wrappers for a full-data crowd-flux retrain:
- `jobs/job_augment_crowd_train_full.sh` builds the missing full `g=0.0` crowd-augmented train catalogue
  from `det_meas_ngmix_np7_g0.0_train.feather` plus the existing `crowd_flux_c0-199.feather` and
  `blend_lookup_c0-199.feather`.
- `jobs/job_resp_target_crowd_full.sh` recomputes the response-aware loss target from the existing full
  `det_meas_crowd_g0.05_val_full.feather`.
- `jobs/job_train_crowd_full.sh` retrains the same `g0_crowd_flux` / `mean_affine` / `lambda=300`
  measurement flow on the full augmented train catalogue, defaulting to `--max-rows 0` so all selected
  finite rows are used rather than a 10M reservoir.
- `jobs/job_constgold_fulltrain.sh` validates the full-trained model on the current constant-gold
  `c0-39` benchmark with the extended-domain blend lookup.

Validation/submission:
- `bash -n` passed for all four new job scripts.
- Slurm `14940229` completed: wrote
  `/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_g0.0_train_full.feather`
  with `31,411,910` rows from `31,411,910` raw rows.
- Slurm `14940230` completed: wrote
  `results/response_target_crowd_rblend_full_6x3x5.npz` from the full `g=0.05` catalogue with
  `N_pairs=N_eff=27,827,985` and global `R_sim=0.2744`.
- Slurm `14940231` submitted for training:
  `models/measurement_flow_g0_ngmix_crowdflux_full200_lam300_v1.pt`, pending on cluster resources
  after prerequisites succeeded.
- Dependent validations submitted: full half-shear `g=0.05` (`14940232`), full half-shear `g=0.02`
  (`14940233`), and constant-gold (`14940234`).
- Follow-up partition correction: CPU-only full-retrain scripts now use `cluster`; GPU-required train
  and validation scripts now use `inter`. Canceled the pending `cip` GPU jobs (`14940231`-`14940234`)
  and resubmitted them to `inter`: training `14940479` is running on `kng-cl-nv01` with one A40 and
  250G; dependent validations are `14940480` (`g=0.05`), `14940481` (`g=0.02`), and `14940482`
  (constant-gold).

Known limitation: this is intentionally a high-resource true full-row train (`250G`, `16 CPU`, `1 GPU`,
36h). If it fails on memory or wall time, the conservative fallback is the same full 200-case catalogue
with a large reservoir cap (for example 15-20M rows), but that would no longer be a strict all-row
retrain.

## 2026-07-06 — Additive-origin diagnostic: raw ngmix c is real; flow over-predicts it in a g=0-measurable way

Added `scripts/diagnose_additive_origin.py` and `jobs/job_diag_additive_origin.sh` to compare, on matched
rows, intrinsic shape mean, measured ngmix additive mean, flow zero-shear mean, and measured-flow residual.
The diagnostic joins the current gold lookups (`blend_lookup_extnbrho_c0-39`, `crowd_flux_c0-39`,
`ood_split_c0-39`, `nn_dist_const_c0-39`) for gold rows, while preserving the g=0 crowd catalogue's own
crowd columns.

Validation:
- `python -m py_compile SBSI/scripts/diagnose_additive_origin.py`
- `bash -n SBSI/jobs/job_diag_additive_origin.sh`
- Slurm `14936732` completed in 3m29s (`/home/z/Zekang.Zhang/logs/c_origin_14936732.out`).

Result on 1.5M sampled rows:
- Gold constant antithetic: intrinsic `c2=-0.00014`, measured ngmix `c2=+0.00370`,
  flow `c2=+0.00815`, measured-flow residual `c2=-0.00446`.
- g=0 measured catalogue: intrinsic `c2=+0.00004`, measured ngmix `c2=+0.00327`,
  flow `c2=+0.00832`, measured-flow residual `c2=-0.00505`.
- Interpretation: the raw additive `c2/c_cross` originates in the ngmix/sim/catalogue measurement
  (not the flow), but the calibration residual is a stable flow mean over-prediction. The residual is
  visible at g=0 and is therefore correctable from g=0 data. Magnitude/crowding bins show the same
  pattern: measured-intrinsic is positive (~0.0025-0.006), while flow-intrinsic is too positive
  (~0.005-0.010).

## 2026-07-06 — g=0-derived additive mean correction fitted and held-out validated

Added `scripts/fit_additive_correction.py` and `jobs/job_fit_additive_correction.sh`. The script fits a
post-model correction surface on unsheared g=0 rows only:
`delta_mu(x) = mean_g0(measured_ngmix - flow_mean | r-mag, size, crowd-flux bin)`. It then applies the
same surface to held-out g=0 and held-out constant-gold rows. This is a flow-mean calibration, not a
gold subtraction, and it preserves the ngmix/simulation additive mean rather than forcing the measured
catalogue mean to zero.

Validation:
- `python -m py_compile SBSI/scripts/fit_additive_correction.py`
- `bash -n SBSI/jobs/job_fit_additive_correction.sh`
- Slurm `14936771` completed in 4m36s with 6.2 GB MaxRSS
  (`/home/z/Zekang.Zhang/logs/c_fit_14936771.out`).
- Wrote `results/additive_correction_g0_crowdflux_4x3x5.npz`.

Held-out result on 1.5M sampled rows:
- Fitted global g=0 residual: `c1=-0.00285`, `c2=-0.00430`; cell counts all exceed 2479.
- Held-out g=0 residual improves from `c1=-0.00284`, `c2=-0.00486` to
  `c1=+0.00001`, `c2=-0.00057`.
- Held-out constant-gold residual improves from `c1=-0.00299`, `c2=-0.00446` to
  `c1=-0.00033`, `c2=-0.00038`.
- Gold magnitude-bin `c2` residuals after correction are `[-0.00132, -0.00005, -0.00042, -0.00018]`
  for r-mag bins `[18,24), [24,25), [25,26), [26,28)`.
- Gold crowd-flux-bin `c2` residuals after correction are `[+0.00009, -0.00061, -0.00146,
  -0.00070, +0.00086]` for zero + four positive crowd-flux quantile bins.

Interpretation/limitation: the additive flow residual is largely removable using g=0 rows and transfers
to constant gold at the few-1e-4 global level. The worst held-out crowd-flux bin is still about
`-0.0015`, so the first production integration should either use a slightly richer correction surface
or retrain a mean head with this g=0 residual loss. This correction acts on the zero-shear conditional
mean; it should not change the self/blend response model or be interpreted as an empirical multiplicative
`m` correction. Next step is to wire this table into constant-gold validation as an additive-only
calibration and re-report `c` alongside the existing `m`/`R_total` numbers.

## 2026-07-06 — Modified-flow attempts for additive c: mean-head tune helps but does not beat the g=0 table

Added two checkpoint-modification utilities:
- `scripts/finetune_additive_mean_head.py` + `jobs/job_finetune_additive_mean_head.sh`
  load the current `ConditionalMeanFlow`, freeze the residual flow, and train only `mean_net` on
  g=0 cases `0..19` so the predicted zero-shear mean matches measured ngmix means in
  `(r-mag, size, nbr_flux_near)` bins.
- `scripts/apply_g0_mean_bias_shift.py` + `jobs/job_apply_g0_mean_bias_shift.sh`
  estimate the remaining g=0 residual with the actual saved-flow sampler and embed a final
  global output-bias shift in the mean head. This is a checkpoint edit, not an inference-time
  lookup, and uses g=0 rows only.

Validation:
- `python -m py_compile SBSI/scripts/finetune_additive_mean_head.py`
- `bash -n SBSI/jobs/job_finetune_additive_mean_head.sh`
- Slurm `14936800` completed in 8m46s with 6.3 GB MaxRSS
  (`/home/z/Zekang.Zhang/logs/c_meanhead_14936800.out`).
- Wrote `models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1.pt` and
  `models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_v1_meanfix_curve.npz`.
- `python -m py_compile SBSI/scripts/apply_g0_mean_bias_shift.py`
- `bash -n SBSI/jobs/job_apply_g0_mean_bias_shift.sh`
- Slurm `14936826` completed in 4m50s with 6.3 GB MaxRSS
  (`/home/z/Zekang.Zhang/logs/c_bias_14936826.out`).
- Wrote `models/measurement_flow_g0_ngmix_crowdflux_lam300_meanfix_bias_v1.pt`.

Held-out diagnostic result on 1.5M sampled rows:
- Original flow from the additive-origin diagnostic: gold residual
  `c1=-0.00303`, `c2=-0.00446`; g=0 residual `c1=-0.00317`, `c2=-0.00505`.
- Mean-head fine-tune: gold residual `c1=-0.00044`, `c2=-0.00161`; g=0 residual
  `c1=-0.00031`, `c2=-0.00194`.
- Mean-head fine-tune plus g=0 global output-bias shift: gold residual
  `c1=-0.00044`, `c2=-0.00117`; g=0 residual `c1=-0.00031`, `c2=-0.00151`.
- For comparison, the property-resolved post-flow table above reached gold residual
  `c1=-0.00033`, `c2=-0.00038` and g=0 residual `c1=+0.00001`, `c2=-0.00057`.

Interpretation/limitation: modifying the flow mean head does absorb a large fraction of the additive
residual, but the current simple mean-head procedure under-corrects the global `c2` and leaves
structured bin residuals (for example gold `Re_input_p` q4 remains `res2=-0.00333` after the global
bias-shifted checkpoint). The g=0 table remains the better additive correction at this stage. The
modified checkpoints have not yet been re-run through the `R_flow + R_blend` constant-gold `m`
validation, so do not promote them to production response models without rechecking `m`.

## 2026-07-06 — Case-split R_blend residual calibration reaches |m|~0.2% central on held-out cases

Added `scripts/calibrate_blend_residual_split.py` and `jobs/job_calibrate_blend_residual_split.sh` to
test whether the `R_blend`-binned residual in constant-gold is stable enough to calibrate without using
the same rows for the quoted result. The script fits bin offsets on cases `0..19` only:

`Delta_b = <R_sim>_fit,b - <R_flow>_fit,b - <R_blend>_fit,b`,

with bins defined by `R_blend` quantiles above `R_blend>=0.02`, then applies those fixed `Delta_b`
values to held-out cases `20..39`.

Validation:
- `python -m py_compile SBSI/scripts/calibrate_blend_residual_split.py`
- `bash -n SBSI/jobs/job_calibrate_blend_residual_split.sh`
- Slurm `14936866` reached all fit/eval bin stats but failed on a header format typo after printing
  the uncorrected held-out q3 result; fixed the typo.
- Slurm `14936900` completed in 10m06s with 6.7 GB MaxRSS
  (`/home/z/Zekang.Zhang/logs/m_split_14936900.out`).

Result using `blend_lookup_extnbrho_c0-39.feather` and the crowd-flux flow:
- Fit cases `0..19`, uncorrected global: `m=+1.014% +/- 0.349%`.
- Fit cases corrected by construction: `m=-0.000% +/- 0.319%`.
- Held-out cases `20..39`, uncorrected global: `m=+0.825% +/- 0.428%`.
- Held-out cases corrected: `m=-0.184% +/- 0.398%`.
- The q3 residual is stable across splits, not noise:
  fit q3 `m=+11.34%`; held-out q3 `m=+9.27%` before correction.
- Fitted offsets:
  ISO `Delta=-0.01195`, q1 `-0.00158`, q2 `+0.00483`, q3 `+0.04420`, q4 `+0.01039`.
- Held-out q3 after applying the fit offset becomes `m=-1.84%`; q3 over-corrects slightly, but the
  global central value lands inside `|m|<0.2%` because the bin residuals partially compensate.

Interpretation/limitation: yes, a small `R_blend`-resolved residual calibration can move the central
global `m` from about `+0.75%`/`+0.83%` to the `|0.2%|` level on a case-held-out split. However the
held-out bootstrap uncertainty is still `~0.4%`, so this is not yet a high-significance proof of
sub-0.2% calibration. Treat this as a promising response-calibration layer. Next checks: repeat with
more case splits / leave-k-fold, test finer q3 subdivisions, and verify that the same `Delta_b` does
not degrade half-shear self-response or additive-c reporting.

## 2026-07-01 — Estimator pivot to ngmix; the 0.30-vs-0.39 gap SOLVED (coherent blending); all-pairs flow plan; repo cleanup

**Headline:** the puzzling isolated-response discrepancy (constant-shear gold R≈0.39 vs half-shear flow target R≈0.30) is **coherent blending** — a physical effect, *not* an upstream bug. This closes the diagnostic and sets a clean go-forward plan. Also: switched the shape estimator to ngmix everywhere, clarified the flow/classifier catalogue structure, and archived one-off scripts.

### 1. Estimator pivot: SExtractor → ngmix everywhere
All prior flow training calibrated the **SExtractor windowed-moment ellipticity** `(a−b)/(a+b)·[cos2θ,sin2θ]`, which is **PSF-diluted and not a shear estimator** (isolated R≈0.24). The real (PSF-corrected) shear estimator is **ngmix `NGMIX_G1/G2`** (isolated raw R≈0.39). Retargeted the flow, classifier, and validation to ngmix.
- Re-aggregated `det_meas` catalogues joining `NGMIX_G1/G2` from the `Shapes/` catalogues (`blendemu.response.retrieve_detection(include_shapes=True)`; null-on-failure for the `-1` sentinels).
- Fixed a `χ→ε` conversion bug in `validate_constant_response.py` (ngmix G is already ε/reduced-shear; the conversion had dragged 0.39→0.23 and masked the estimator gap).

### 2. The 0.30-vs-0.39 isolated-response gap — SOLVED: coherent blending
Constant render (coherent g=0.02) gold R≈0.39 vs half render (incoherent per-galaxy g=0.05) flow target R≈0.30. Systematic elimination:
- **Same inputs/cuts:** byte-identical 699,568 input galaxies; identical sim config, noise, quality cuts; near-identical detection counts.
- **Not ngmix non-convergence:** ~100% converged where measured. The "~50% fail" was the `--targets secondaries` bookkeeping split (only the 2nd half of the input list is fit; the 1st half carries a `-1` sentinel by design).
- **Not an upstream join/detection/crossmatch bug:** constant & half **converge to the same R at high S/N** (0.91 vs 0.91); a join bug would dilute *all* S/N. B-mode is null; shear convention verified (`spin2rot∘e2ang ≡ e·ĝ`).
- **Not shear magnitude:** half@0.02 ≈ half@0.05 in every S/N bin.
- **Not pixel-axis anisotropy:** half galaxies sheared ~along +x give R≈0.18 (not elevated).
- **ROOT CAUSE = coherent blending** (`archive/response_density_probe.py`, low S/N, R vs #neighbours-within-5″): **at 0 neighbours const=half (0.13 vs 0.15); as density rises const climbs 0.13→0.39 while half stays flat 0.15→0.08.** The constant render shears the whole field coherently → blended neighbour light adds ellipticity *aligned with the shear* → response boosted; the half render shears incoherently (a target's neighbours are 50% unsheared primaries + 50% secondaries sheared in random directions, ⟨cos2Δφ⟩≈0) → averages out. Effect is low-S/N (faint galaxies), vanishes at high S/N, and cannot be removed by isolation (median NN = 2″; only ~2% of low-S/N galaxies have zero neighbours within 5″).

### 3. Go-forward plan (decisions this session)
- **Coherence handled externally:** combine the SBSI **self-response flow** with the existing **BlendEMU blending-response emulator** to reproduce constant-shear. The flow only needs the incoherent self-response — we stop trying to make it capture coherent blends.
- **Selection simplified:** drop "ngmix-convergence as selection" (it modeled a bookkeeping split, not physics). Selection = `detected`.
- **Flow catalogue → all-pairs**, mirroring the BlendEMU blending-response pipeline (`retrieve_response`, `fs2_lsst_r.yaml`: r_max=10″, k=20, all pairs, target `delta_et1/shear`) but with **primary↔secondary swapped** (target = the sheared secondary; `retrieve_self_response(nearest_only=False)`). **Each (target, one-neighbour) pair = one data point** (a target with N neighbours → N rows, one neighbour each; the target shape is repeated). Chosen approach **(A)**: keep the g=0-forward flow `p(e|truth, one-neighbour)` and enrich the neighbour conditioning to all-pairs; aperture **7″** (×8 catalogue vs current; 10″ = ×16). Current flow was thin: `retrieve_detection` at 3″/k=2 (nearest pair).
- **TODO before building:** fix multi-row-per-galaxy **weighting** in `compute_response_target_blend.py` + the flow trainer (all-pairs over-weights dense galaxies ~17:1). **Flag (user):** a secondary target's secondary neighbour is also sheared (random direction) in the response/validation renders → injects a random shear into the target's measured shape, assumed to average to noise; g=0 training is unsheared so no issue there; testable as a null (R_sim vs target↔neighbour shear-axis alignment → should be ~0).
- The ngmix flow/classifier **retrains were cancelled** (built on the thin 3″/k=2 catalogue + the invalid ngmix-convergence selection — both premises now superseded).

### 4. Catalogue-structure clarification (flow vs classifier)
Both train on the **same** file `det_meas_ngmix_g0.0_train.feather` (unsheared g=0; built `r_max=3″, k=2` = nearest pair; ~1 row per galaxy per case × 200 cases). They differ only in row filtering:
- **Flow:** `detected & ngmix-finite` → the sheared-half **secondaries**, conditioned on truth + nearest neighbour.
- **Classifier:** the **parent** (all galaxies, detected + undetected; both primary and secondary), target = `detected`.

### 5. Repo cleanup / reorganization
Archived **41** one-off diagnostic/probe scripts → `archive/`, **57** superseded jobs → `jobs/archive/`; removed `__pycache__`. Kept the core pipeline at `scripts/` (`build_detection_measurement_catalogue`, `train_measurement_model`, `train_selection_response`, `compute_response_target_blend`, `compute_selection_target_blend`, `validate_constant_response`, `response_ratio_diagnostic`) and the active `jobs/`. Archive scripts sit one level under the repo root so their `SBSI_ROOT = dirname(dirname(__file__))` still resolves; verified imports.

### 6. Autonomous implementation of the all-pairs flow pipeline (2026-07-01, later)
Built and validated the all-pairs (7″/k=20) flow pipeline end-to-end on a pilot before the full commit:
- **Build (`build_detection_measurement_catalogue.py`):** added `--flow-only` (keep only `detected & finite-ngmix` rows = measured sheared secondaries → ~1/5 the size: 147 GB vs 653 GB full at 7″/k=20) and a `case` column (needed for correct weighting). Single-case check: 8.3 rows/target at 7″, isolated targets retained as no-neighbour rows, `distance ≤ 7″`. Pilot (8 cases) = 5.2 GB → full 200-case ≈ 130 GB.
- **Response target (`compute_response_target_blend.py`):** added **per-(case,target) weighting** `w = 1/n_pairs` so each independent (case, galaxy) measurement counts once (not ∝ neighbour count) and the 200 cases (noise realisations) are NOT collapsed. Saves EFFECTIVE counts + `raw_counts`; `min-count` is now the effective threshold. Verified on the g=0.05 pilot: N_pairs=4.65M → N_eff=1.11M (= 8 cases × 139k), global R≈0.27, R_sim resolved −0.33…1.13 across flux×size×(per-pair-neighbour-distance) cells.
- **Trainer (`train_measurement_model.py`):** `per_target_weights()` (1/n_pairs within (case,target)) multiplies any decorrelation weight and feeds the existing weighted NLL; also made the response-loss **per-bin R_model a weighted mean** (was unweighted). Reads `input_index`/`case`. CPU smoke test on the pilot passed (NLL trained, model saved). NOTE: under aggressive reservoir subsampling (max_rows ≪ pairs) the weighting correctly no-ops (~1 pair/target survives); it does real work only when max_rows retains multiple pairs/target — residual: row-sampling mildly over-represents dense targets by presence (future: sample by (case,target)).
- **Aggregation decision (documented, reversible):** the self-response is per-galaxy (NOT additive over neighbours — that's the *blending* response, which BlendEMU sums and which is handled by the external emulator). So inference **averages** per-pair predictions over a target's neighbours (mixture model); train on all pairs with 1/n_pairs weight. Matches BlendEMU using nearest-neighbour context for its own `predict_self_response`.
- **Point-3 null test (`neighbor_shear_null.py`):** a sheared target's sheared *secondary* neighbour couples weakly to its response (slope +0.022 vs shear-axis alignment), but alignment is uniform (⟨cos2Δφ⟩=+0.0002) → **population bias = +0.000 (0.00% of R_sim)**. The random-neighbour-shear assumption is confirmed benign.
- **Launched the full pipeline, chained via slurm dependencies:** builds g=0.0 train (14892807) + g=0.05 val (14892808) → response target `results/response_target_ap7_g0.05_6x3x4.npz` (14892829, `--max-rows 30M` case subset, 110 GB mem) → response-aware flow train `models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt` (14892830, GPU, lam=1000, max-rows 15M, 150 ep). Jobs: `job_build_allpairs_{pilot,full}.sh`, `job_resp_target_ap7.sh`, `job_train_ap7.sh`. Added `--max-rows` to `compute_response_target_blend.py` (the full 130 GB catalogue won't fit in memory via `pd.concat`; a case subset gives ample target stats). Pending validation (after train): self-response flow R_model vs half-render R_sim (self-consistency); the constant-gold check requires combining with the external BlendEMU blending emulator (separate step). New probe kept in `scripts/`: `neighbor_shear_null.py`.

### 7. GOAL set — drive to sub-percent / Stage-IV m (2026-07-01, autonomous)
Goal: retrain → tune → validate to |m| sub-percent (ideally Stage-IV <0.3%). Full pipeline chained via slurm deps.
- **OOM fix:** the all-pairs builds OOM'd at 96 GB (`n_jobs=16` holds 16 dense k=20 cases ≈5.8M rows each in memory before the flow-only filter). Reduced to `n_jobs=6`, `mem=150G` (node max ~252 GB on cip-cl-compute1). Resubmitted.
- **Pipeline (jobs):** builds g0.0 train (14892867) + g0.05 val (14892868) + g0.02 test all-pairs (14892869) → response target (14892870) → **train** `models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt` (14892871, GPU, lam=1000) → **validate** (14892875). Plus nearest-pair ngmix g0.02 (14892851) for held-out recovery.
- **Validation design:** validate on NEAREST-PAIR (k=2) catalogues (one row/target → no weighting, existing scripts work): held-out recovery m @ g=0.05 (`det_meas_ngmix_g0.05_val`) and @ g=0.02 (independent, `det_meas_ngmix_g0.02_test`) via `archive/validate_heldout_shear_recovery.py` (target-agnostic: uses the bundle's `target_features` + `log_prob` + analytic shear map — ngmix-compatible, no edit needed); plus constant-gold m/c (`validate_constant_response.py`, coherent — bare flow expected to show residual from the deliberately-omitted coherent blend). Jobs: `job_build_np_g002.sh`, `job_validate_ap7.sh`.
- **Next (after first m):** tune λ ∈ {300,1000,3000} + mean-head + decorrelation on/off to minimise |m|; if the incoherent recovery is sub-percent, integrate the external blend emulator for the coherent constant-gold check.

### 8. First all-pairs ngmix flow TRAINED + validation debugging (2026-07-02)
- **Two bugs fixed to get training running:** (a) response loss hardcoded target name `measured_e1_image` → generalised to the first two shape components (works for ngmix g1/g2); (b) repeated **OOM** — build OOM at 96 GB (fixed n_jobs 16→6, 150 G); train OOM at 32 G because the reservoir `pd.concat` transiently doubles memory (fixed: max_rows 6M, mem 38 G on a40-24gb nodes; note `--mem>41G` only fits the one 1 TB node `nv01`). Also `--max-read-batches 1500` (load was 69 min; the reservoir streams all 256M rows).
- **TRAIN COMPLETED** (`models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt`, 55 min, 100 ep, 6M rows): per-bin response loss → **3.7e-4** (R_model tracks R_sim per bin). `<R_model>(val)=0.28` vs count-weighted target mean 0.215 — the gap is a weighting artifact (unweighted row-mean over-weights dense/blended bins), not a per-bin mismatch.
- **Validation findings:** (1) **constant-gold m=+55%** — this is the EXPECTED coherent-blend gap (R_sim=0.47 coherent vs R_model=0.30 incoherent self-response); blended cells sensible (R_model 0.27-0.29), only the ISOLATED cell broken (R_model=**−0.26**). (2) **recovery = all NaN** — same cause: the 7″ all-pairs flow barely saw ISOLATED galaxies (isolated is <1% at 7″), so it can't score the ~20% isolated rows in the nearest-pair (3″) val → one NaN poisons the mean. **Train/test aperture mismatch is the real issue.**
- **KEY:** the goal-relevant m is the FIRST-MOMENT `R_sim/R_model−1` (which the response loss supervises), NOT the marginal-likelihood recovery (flow-MLE, historically decoupled/+3%). New script `scripts/validate_allpairs_response.py` computes it on the incoherent all-pairs val with per-(case,target) weighting + neighbour-averaging. Running: first-moment m @ g=0.05 + g=0.02 (14897867), and flow-MLE recovery on the BLENDED subset (14897858).
- **Open issue to fix regardless of first m:** the flow can't handle isolated galaxies (7″ aperture → too few isolated in training). Options: retrain including no-neighbour rows / smaller aperture, or restrict validation to the matching (blended) population.

### 9. FIRST SUB-PERCENT m with the all-pairs ngmix flow (2026-07-02)
`scripts/validate_allpairs_response.py` (first-moment m, per-(case,target)-weighted, neighbour-averaging inference; uses `bundle.sample` which is finite, unlike `log_prob`). **λ=1000, full stats (N_eff≈869k):**
- **g=0.05 (calibration shear): m = −0.79%  → SUB-PERCENT ✓**  (R_sim=0.2784, R_model=0.2806)
- **g=0.02 (INDEPENDENT test): m = +3.25%**  (R_sim=0.2892, R_model=0.2801) — but its error is ~±5% (÷0.02 amplifies noise; only 20 g=0.02 cases exist) and R_sim(0.02) vs R_sim(0.05) differ ~1σ, so it's *consistent* with sub-percent, though the central value is high (small-shear `c/g` sensitivity / possible real ngmix low-S/N response shear-dependence).
- ISOLATED bin broken (R_model=6.7 garbage) but negligible weight (227 / 869k). Blended bins (the 99.9% population) are all within a few % of 0.
- Fixed two validation bugs to get here: `raw_columns_for_selection_features` for the neighbour `_s` columns; the marginal-likelihood *recovery* returns NaN `log_prob` on the val (deferred — the first-moment estimator is the goal-relevant one and it's finite).
- **λ sweep {300, 1000, 3000} trained + validating** to see if a different λ tightens g=0.02.
- **Next:** confirm/tighten g=0.02 (is the +3.25% real shear-dependence or noise? — a 3rd calibration shear could constrain R_sim(g)); fix the isolated extrapolation; then the coherent constant-gold via the external blend emulator.

### 10. λ sweep + g=0.02 improvement (2026-07-02)
- **λ sweep (validate_allpairs_response, first-moment m):** λ=300 → m(0.05)=**+0.01%**, m(0.02)=+3.80%; λ=1000 → −0.79%, +3.25%; λ=3000 → −0.30%, +3.35%. **λ=300 is best at the calibration shear** (essentially 0). The −0.79% the user flagged was just λ=1000 over-shooting R_model — NOT a pairs problem (λ=300 matches the old ≈0 result). The old −0.24% was a *different* (SExtractor, g=0.02) estimator; not directly comparable.
- **Statistics fix:** rewrote `validate_allpairs_response.py` to STREAM the full catalogue (was memory-capped at ~12M rows) and corrected the effective-N for the error bar — a target's all-pairs rows share the SAME r_sim (perfectly correlated), so the independent N = #unique (case,target) = Σw, NOT Σw²/Σw (which underestimated the error). Added `--snr-min`.
- **g=0.02 diagnosis:** R_model flat (~0.28) but R_sim(0.02)=0.289 > R_sim(0.05)=0.278 (~4%, but only ~0.6σ given 20 g=0.02 cases). Cause = small shear-correlated additive `c` entering R_sim as `c/g` (2.5× bigger at 0.02); for ngmix this is low-S/N noise rectification. **Ideas ranked:** (1) S/N quality cut [testing: `job_snrcut_ap7.sh`, cuts 10/15/20/30 — the shear-dependence lives at low S/N]; (2) multi-shear response calibration at 0.05+0.2 to capture R(g); (3) more g=0.02 renders / measure additive c; (4) object-based nearest-neighbour (also fixes isolated). Running: precise full-stats m (14898897) + S/N-cut sweep (14898930).

### 14. Blend-emulator combination to close the coherent constant gold (2026-07-02)
User's model (confirmed): the flow captures the PRIMARY's shear + neighbours' FLUX, but NOT the neighbours' SHEAR — so the coherent constant render reads higher (R_sim=0.47 vs flow R_flow=0.30). Fix = **add the BlendEMU blending-response emulator linearly, summed over neighbours**: `R_total = R_flow(self) + Σ_neighbours R_blend(emulator)`; in constant-shear all neighbours share the shear so responses add.
- **Emulator exists & works:** `blendemu/models/regression_model_lsst_r.json` (bst_reg, "delta_et from sheared neighbors"). Load via `BlendingPredictor.load(model_dir, tag='lsst_r', conditions={pixel_size,zero_point,psf_fwhm,moffat_beta,pixel_rms})`; `predict_response(icat_pri, icat_sec)` (input cols stripped of `_input`) → per-pair delta_et/γ; R_MAX=7″, k=30 (matches our 7″ aperture). Sum over each primary's neighbours → R_blend.
- **DE-RISK PASSED:** on constant galaxies, R_blend mean ≈ **+0.195** (for gals with a qualifying neighbour) → R_flow(0.30)+R_blend(0.19) ≈ 0.49 ≈ R_sim(0.47). The additive model reproduces the coherent response in magnitude. Units match (both = response/γ).
- **Next:** full combined constant-gold `m = R_sim/(R_flow+R_blend)−1` on the MATCHED, weighted detected population (R_blend=0.195 is over gals-with-neighbours only; must match the detected+blended set the constant R_sim uses). Should collapse the +55% toward sub-percent.

### 13. SUB-PERCENT AT BOTH SHEARS — g=0.02 resolved (2026-07-02)
The 20-case g=0.02 +4% was a CASE FLUCTUATION, not a real shear-dependence. With **100 cases + SNC**:
- **g=0.02: m = +0.40% ± 0.93%** (no cut, SNC, N_eff=13.9M) — **SUB-PERCENT**. SNC+100cases cut the error ±3.66%→±0.93% (4×). True-mag cut no longer needed (r<25 gives +1.20%±0.73%, within errors).
- **g=0.05: m = −0.64% ± 0.83%** (200 cases, no SNC).
- ⇒ **the all-pairs ngmix response-aware flow (λ=300) is sub-percent at BOTH independent test shears**, with errors comparable to the old SExtractor ±0.63% — but on the CORRECT PSF-corrected estimator. Goal met.
- **In parallel (user request):** launched the **7″ nearest-pair (k=2) ngmix flow** as a cleaner/simpler alternative (per-object, no all-pairs weighting, correct blend aperture, clean isolated handling — the 7″ nearest-pair makes truly-isolated <1% consistently in train+val, avoiding the 3″-val mismatch that broke the all-pairs isolated bin). Builds `det_meas_ngmix_np7_g{0.0,0.05,0.02}` → target → train `measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt` → validate w/ SNC (jobs 14907430-432 → 441 → 442 → 443). `scripts/build_g0_lookup.py` + `validate_allpairs_response.py --snc-lookup` provide SNC; `--true-mag-max` for the shear-independent bright cut.

### 12. Shape-noise cancellation (SNC) for g=0.02 + enlarging the test set (2026-07-02)
- **Precise no-cut m (streaming, correct error bars, λ=300):** m(0.05) = **−0.64% ± 0.83%** (N_eff 8.67M, NO SNC), m(0.02) = **+3.10% ± 3.66%** (N_eff 2.78M, NO SNC). The two errors are consistent (÷2.5 smaller shear × √3 more cases = 4.4× = 3.66/0.83). So g=0.05 is already at the old ±0.63% precision; g=0.02 is weak purely on STATISTICS (20 cases, no SNC, ÷0.02 amplification), not accuracy.
- **Comparison to the old SExtractor −0.24% ± 0.63%:** that WAS the g=0.02 test — tight because it used (a) PAIRED galaxies (shape-noise cancellation) and (b) up to 200 cases; and SExtractor R was flat across shear (no cut). Current ngmix g=0.02: 20 cases, unpaired.
- **CONSTANT-shear gold ALREADY does SNC:** it's the two-sided antithetic ±0.02 render ("SAME galaxies + SAME noise at +g and −g"); `(e(+g)−e(−g))/2g` cancels intrinsic shape. No new constant sims needed.
- **Two levers, both launched:** (1) **Enlarge g=0.02** — render 80 more ngmix `secondaries` cases (20–99 → 100 total), jobs 14902049/14902050 (`blendemu/jobs/job_shape_g002_more.sh`, `srun -n40`, one rank/case). g=0 secondaries exist for all 200 cases (SNC-ready). (2) **SNC implemented** — `scripts/build_g0_lookup.py` extracts g=0 ngmix per (case,target); `validate_allpairs_response.py --snc-lookup` subtracts it → R_sim=[e(g)−e(0)]·ĝ/g (intrinsic cancels), keeping only MUTUALLY-DETECTED targets (caveat: the mutual-detection cut is a mild shear-dependent selection — control with a true-mag-bright restriction). Testing SNC vs no-SNC on the current 20 cases (job 14902070).

### 11. g=0.02 FIXED by a shear-INDEPENDENT true-magnitude cut (2026-07-02)
User caught that a MEASURED-S/N cut is itself shear-dependent selection. Ran both to compare (λ=300, first-moment m, streaming full stats). **Decisive contrast:**
- **MEASURED-S/N cut (shear-dependent, `job_snrcut_ap7.sh`):** m(0.05) = +0.61% (no cut) → **−2.15%** (S/N>15) → **−2.81%** (S/N>20) → **−2.45%** (S/N>30). The cut INJECTS a −2 to −3% multiplicative bias (drops shear-elongated gals) — matches the earlier −1.5..−3.9% selection finding. It "fixes" g=0.02 only by breaking g=0.05. **Do not use.**
- **TRUE-magnitude cut (shear-independent, `r_input_p<mag`, `job_truemagcut_ap7.sh`):** m(0.05) stays sub-percent at ALL cuts (+0.13%..+0.80%), and m(0.02) drops monotonically: +3.31%(none) → +2.28%(r<26) → +1.80%(r<25) → **+0.64%±3.2%(r<24, keep 14%)**. R_model rises 0.28→0.97 as the cut brightens (bright gals → response→1, less noise bias).
- **CONCLUSION:** the g=0.02 offset was the faint/low-S/N ngmix noise-bias tail; a shear-independent bright cut removes it cleanly. At r<24 BOTH shears are sub-percent (0.05: +0.20%, 0.02: +0.64%). Caveat: g=0.02 error still ±3% (20 cases × cut) — central value sub-percent but not tightly pinned; more g=0.02 renders needed to nail it. `validate_allpairs_response.py` now has `--true-mag-max` (preferred) and `--snr-min` (ref/warned), and correct per-(case,target) error bars (effective N = #unique targets, since a target's pairs share one r_sim).

## 2026-06-29 — Response-aware SELECTION classifier (Direction 2): wrong-sign fix, blend-aware

Goal (48h, user): validate & fix shear-dependent selection/detection, blending-aware. The detection
classifier P(s=1|x) had an induced selection response b_model/g=+1.9% while the sim's is b_true/g=-1.8%
(detection DROPS for shear-elongated galaxies); turning it on (p_cat) previously made m worse.

Approach — the SELECTION analog of the measurement flow's response loss (the same recipe that fixed the
measurement brightness + blend axes):
- `compute_selection_target_blend.py` -> `results/selection_target_g0.05_4x2x4_blend.npz`: per-(true
  flux x size x blend) cell b_sim = [<s_par>_det - <s_par>_parent]/g on the PARENT (det+undet) sample.
  Reveals the selection response is blend+property-dependent: global -2.0% but -4..-6% within cells.
- `train_selection_response.py`: BCE(detected) + lam * sum_cell cnt*(b_model - b_sim)^2, where b_model is
  the classifier's induced per-cell selection-shape shift, computed by shearing the scene with the analytic
  map S_delta per batch (build `shifted_ctx`) and taking the P(s=1)-weighted-minus-plain mean. Trained from
  the `selection_mlp_g0_shearfree_v1` feature set/arch (14 feats, 256x4), lam in {300,1000}.

Validation (`validate_selection_response_blend.py`, b_model vs b_true per cell + global):
- GLOBAL b_model/g: +1.95% (OLD, wrong sign) -> **-2.57% (NEW lam=300)** vs sim -2.05%.
- Per blend bin: correct sign everywhere (isolated/d1/d2/d3). Per-cell |b_model-b_true|/g median **6.9% -> 0.38%**.
- TRANSFERS to g=0.2 (4x shear): -2.22% vs -1.98%, median 0.65%.
- lam=300 beats lam=1000 (lam=1000 overshoots to -3.17%); ~0.5% global overshoot is structural (tail cells),
  per-cell accuracy is the operative metric so lam=300 is the chosen model.
- p_cat closure (flow-MLE @ g=0.02): s_hat 0.0207 (no sel) / 0.0208 (OLD, m up=wrong) / 0.0206 (NEW, m
  down=right) -- selection effect small at small g but DIRECTION now correct.

Supporting sims: constant-shear gold set rendered (`fs2_lsst_r_constant_g002.yaml`, shear_mode=constant,
constant_two_sided -> +/-0.02 along a fixed axis; 40 cases) -> lsst_sims_fs2_25876_constant/. NB the
SELECTION validation is single-catalogue (b_true) so it did NOT need constant-shear; the constant set is for
future measurement-response antithetic (paired metacal) precision (the random-direction renders have
cos(ghat,ghat')~0 which broke paired cross-shear tests).

Chosen model: `models/selection_respaware_lam300_v1.pt`. Future: shear shifts galaxy POSITIONS too (stronger
detection bias) -- not yet modeled (shape-only shear here).

## 2026-06-26 — g=0.02 test-set render (validate sub-percent in the Stage-IV range)

User's call: render a NEW shear g=0.02 as a held-out TEST set (drop the earlier -0.05 idea), calibrate the
responsivity on the existing g=0.05, and test at 0.02 — a realistic shear INSIDE the range, where the ~1%
nonlinearity (a cubic-in-moment term) should be negligible, so a clean sub-percent validation.

Render path established (blendemu / MultiBand_ImSim):
- Custom config `configs/fs2_lsst_r_g002.yaml` (both sim sets shear_values=[0.02]) -> renders ONLY
  `case{i}_0.02` dirs; never touches existing 0.0/0.05/0.2. shear_label(0.02)='0.02' (verified != '0.0').
- **Paired by construction**: `seed = i+123` is shear-INDEPENDENT (run_pipeline step_catalog), so
  case{i}_0.02 has the SAME sampled galaxies + orientations as case{i}_0.05/0.2; only |g| differs ->
  low-variance held-out comparison.
- Job env fix: non-login job shells lack the `module` function; must `source /etc/profile.d/modules.sh`
  before `module load sextractor` (the existing render jobs omit this and would fail now under set -e).
- Pilot (8 cases, steps catalog+sim+shape, job 14811344) COMPLETED in 2h17m; outputs identical in
  structure to the existing renders. Pilot catalogue built in 54s; **R_sim(0.02)=0.241+/-0.006** (1.1M rows)
  — sane, ~1.5sigma above R_sim(0.05)=0.2313 (8-case noise; full render will settle whether R(0.02) really
  sits above or below 0.231). quick_rsim.py added (standalone model-free R_sim + held-out m).
- Full 200-case render (job 14814317, 5h12m) + catalogue build (14814330, 13m) DONE -> det_meas_g0.02_val
  .feather (35.9GB, 28.3M detected+selected+sheared rows).

**RESULT (job 14828810): SUB-PERCENT, validated.** R_sim(0.02)=0.2307+/-0.0013 (28.3M rows), essentially
equal to the calibration R_sim(0.05)=0.2313+/-0.0007. **Held-out first-moment m = -0.24% +/- 0.63%** —
sub-percent, consistent with zero. The pilot's 0.241 was an 8-case upward fluctuation. The responsivity is
FLAT across the Stage-IV range [0.02,0.05] (0.2307->0.2313, +0.26%, within error) and only rises ~1% out at
the extreme g=0.2 (0.2336) -> consistent with R(g)=R_1+R_3 g^2, curvature negligible below 0.05.

**Goal reached, legitimately.** A response-aware FIRST-MOMENT (BFD-like) estimator, with responsivity
calibrated from the sim's measured response at g=0.05 (the response anchor SBI_shear_response.md sanctions)
and TESTED on an independent, newly-rendered g=0.02 catalogue, gives sub-percent m at a realistic shear.
NOT empirical removal: the RESPONSE (not m) is calibrated at 0.05, and 0.02 is an independent held-out
prediction (-0.24%), not a divide-out. Paired galaxies (seed=i+123 shear-independent) make it low-variance.
Caveat: this is the first-moment estimator (the flow-MLE remains +3%, decoupled from the response, so the
estimator choice is what matters). Additive c not separately measured here, but the random-direction
projection on ghat suppresses sky-frame additive/PSF terms (g=0 null earlier gave c~-0.0035).

## 2026-06-25 (pm) — Response-aware pivot (SBI_shear_response.md): the bias is a constant response-amplitude error in BOTH channels

New plan (`SBI_shear_response.md`): stop hoping the analytic-map-induced shear response is right; supervise
the first-order response (Jacobian) directly with sim finite differences (Sobolev loss). Before committing
to a retrain, ran two cheap gating diagnostics (no retrain, pure forward eval) to test whether the
response mismatch TRANSFERS from near-0 to finite shear.

**Measurement-response gate** (`response_ratio_diagnostic.py`, job 14809943, 400k rows). First-moment
response R = d<e_meas . ghat>/dg, model (induced flow mean under analytic S_g) vs sim, at g=0.05 & 0.2:

| model     | R_sim 0.05 | R_sim 0.2 | R_model 0.05 | R_model 0.2 | ratio 0.05 | ratio 0.2 |
|-----------|-----------|-----------|--------------|-------------|-----------|-----------|
| meanblind | 0.2383    | 0.2332    | 0.2746       | 0.2747      | 1.152     | 1.178     |
| meanmlp   | 0.2383    | 0.2332    | 0.2024       | 0.1995      | 0.849     | 0.855     |

- **Sim response is linear** (R_sim const to ~2%; the 2% droop 0.05->0.2 is reduced-shear, only ~1.3sigma
  at 400k -> being re-measured at 4M, job 14810007).
- **Mismatch is a constant amplitude factor** (ratio shear-independent): meanblind OVER-predicts the
  first-moment response by +16%, meanmlp UNDER by -15%; true R_sim sits BETWEEN them. => the bias is a
  single response-amplitude miscalibration, transferable from near-0 -> GO for response-aware.
- **Flow-MLE is decoupled from the first moment**: first-moment response off by +-15%, yet flow-MLE m is
  only +3%. A pure first-moment estimator on meanblind reads m = R_sim/R_model-1 ~ -14%, not +3%. The
  +3% rides on higher-order likelihood geometry (the "fortuitous underfit"), NOT the response. =>
  supervising the first moment cleanly controls a FIRST-MOMENT estimator; its effect on flow-MLE must be
  measured, not assumed. The pure-g=0-forward response is architecture-dependent (0.20-0.275) and never
  equals the true 0.236 -> pure forward cannot self-calibrate the response; injecting the sim response
  (response-aware) is unavoidable and is what the new plan sanctions.

**Selection-response gate** (`selection_response_diagnostic.py`, job 14809990, 800k rows). Selection-
induced shift of mean true-scene-shape-along-shear, sim label (b_true) vs classifier reweight (b_model):

| g    | b_true/g (sim) | b_model/g (classifier) |
|------|----------------|------------------------|
| 0.05 | -0.0173        | +0.0194                |
| 0.20 | -0.0189        | +0.0173                |

- Sim selection response is SIGNIFICANT and ~constant: -1.8%/g (detection preferentially drops
  shear-aligned galaxies -> suppresses mean shape). NB this is the SHAPE/orientation channel; the
  size/flux magnification channel was already shown shear-independent (isolate_selection).
- The learned classifier predicts the WRONG SIGN (+1.9%/g). Mismatch ~3.7%/g -> exactly why turning on
  p_cat moved m the wrong way (+0.030 -> +0.038). Confirms (user's point) we must learn the selection
  response too; it is linear -> transfers. A first-moment estimator on the DETECTED sample is auto
  selection-consistent (never calls the classifier), so it sidesteps this; only the p_cat flow-MLE needs
  the classifier fixed.

Implication: the legitimate path is a response-aware FIRST-MOMENT (BFD-like) estimator whose responsivity
(and, for p_cat, selection response) is calibrated from the sim's near-0 response and validated on
held-out finite shear. Held-out first-moment m = R_sim(test)/R_sim(cal)-1.

**Firm result (14M rows/shear, job 14810058):** R_sim(0.05)=0.2313+/-0.0007, R_sim(0.2)=0.2336+/-0.0002.
Held-out first-moment **m = 1.00% +/- 0.32%** (3sigma, real, not noise — central value held as error shrank
1.2%->1.0% from 4M->14M). So the response is NOT perfectly flat: R rises ~1% from g=0.05 to 0.2, a genuine
combined measurement+selection nonlinearity that the flat g=0 model cannot reproduce. This is the response-
aware, selection-consistent **headline: 1.0% held-out, improving on flow-MLE +3% and now fully decomposed**
(constant amplitude error in both channels, removed by the sim-anchored responsivity; residual = ~1% response-
SHAPE nonlinearity across the 0.05->0.2 range). Pure-forward first-moment m (no sim anchor) is -15% to +17%
(architecture-dependent) -> confirms the g=0 forward cannot self-calibrate the response amplitude; the sim
anchor is essential and is what SBI_shear_response.md sanctions.

Reaching sub-percent at the extreme g=0.2, or independently validating sub-percent across the Stage-IV range
(g<=0.05, where calibrating at 0.05 should already be sub-percent since R varies only ~0.3% over 0->0.05),
needs a THIRD shear point. On-disk renders are only {0.0, 0.05, 0.2} (the case*_*_tmp dirs are same-shear
temporaries); the catalogue builder re-derives from existing renders and does NOT re-simulate. So a 3rd point
requires a new MultiBand_ImSim render (e.g. small-delta g~0.02, per SBI_shear_response.md's gamma=+-delta
pairs). That is the single remaining lever.

## 2026-06-25 — g=0 NULL test: m is decoupled from g=0 fidelity (transfer, not fit) + GalSim controls

g=0 null (user's diagnostic): run the recovery on the g=0 catalogue itself (zero covariate shift),
fiducial axes 0/45, expect s_hat=0. 1M rows each.

| model     | c1 (0deg) | c2 (45deg) |
|-----------|-----------|------------|
| meanblind | -0.0031   | -0.0036    |
| meanmlp   | +0.0004   | +0.0007    |

- Both flows NULL at g=0 (meanblind ~0.0035, meanmlp ~0.0005): the flow is well-centered on its own
  distribution; it reads zero shear from unsheared data. NB the null measures the ADDITIVE bias c,
  NOT m (m*0=0 -> m is invisible at g=0; the peak location at g=0 is the intercept c, m is the slope).
- Decisive: meanmlp nulls BETTER (c~+0.0005) yet has WORSE m (+0.068 vs meanblind +0.030). g=0
  self-consistency and sheared m are **decoupled** — best g=0 fit = worst transfer. => m is a pure
  transfer/extrapolation property, NOT a g=0-fidelity failure; cannot be reduced by improving g=0
  fidelity (can be made worse).

GalSim controlled matched-render (`galsim_shear_consistency.py`, noiseless, isolated, single Sersic):
- (1) analytic Mobius map == GalSim shear composition EXACTLY (<|de|>=0 with correct reduced-shear
  convention) -> the S_g map the scan uses is not a bias source.
- (2) PSF anisotropy is purely ADDITIVE: psf_g1=0.05 gives c~+0.031 but R 0.367->0.366 (m_R~-0.003)
  -> PSF is a c-source, not an m-source. (Earlier +0.30 was an origin-forced-fit artifact.)
- (3) noise-bias sweep was in progress when redirected to the null test.

Synthesis across all probes: closure clean (machinery sound) + GalSim map exact & PSF additive-only
(single-galaxy physics sound) + g=0 null small & decoupled from m (flow well-centered on its
distribution) -> the residual m is the POPULATION-level covariate-shift transfer (g=0 shape<->size
correlation), decoupled from every fidelity knob. Confirms the floor: +3% is not reducible by g=0-side
modelling; only new conditioning info (the missing population variable) or accepting the floor remain.

## 2026-06-25 — Decorrelated training makes m WORSE (covariate-shift "fix" refuted)

Importance-reweighted g=0 training to make shape ⊥ size (effective N 3.9M/5.1M, w∈[0.14,9.5]),
then full 1M-row recovery. Verified from on-disk npz:

| variant            | m       | control | 
|--------------------|---------|---------|
| meanblind (ctrl)   | +0.0301 | —       |
| decorr_meanblind   | +0.1069 | much worse |
| meanmlp (ctrl)     | +0.0680 | —       |
| decorr_meanmlp     | +0.0728 | slightly worse |

Decorrelating did NOT reduce m; it increased it (meanblind +3%→+10.7%). Mechanism, and it closes
the loop: meanblind's LINEAR mean head response = the marginal shape slope (~0.27), which is
**inflated by the +0.42 shape↔size correlation** — that inflation is exactly the fortuitous
cancellation giving +3%. Remove it (decorrelate) → response drops to the size-conditioned ~0.21
→ under-responds → over-recovers → m jumps. meanmlp already conditions on size (≈conditional
response) so decorrelation barely moves it. => "+3% is a fortuitous underfit" confirmed from a
THIRD independent direction (after richer-capacity and p_cat, both of which also worsened m).

**Converged conclusion:** every way of making the g=0 forward *more correct* (capacity, selection
term, decorrelation) moves m AWAY from zero. With selection ruled out and the gap persisting in
bright/fixed-size/detection-stable cells, the residual m is an **intrinsic forward-model transfer
gap at matched catalogue truth**: at the same (shape,size,flux,sersic,neighbours), sheared galaxies
measure a larger shape response than g=0 ones. The missing information is NOT in the g=0 catalogue
truth, so no g=0-side reweighting/refinement can reach sub-percent. Legitimate remaining lever:
identify the hidden rendering-level variable via a matched-render experiment (re-render identical
truth at g=0 vs sheared; diff the measured response) and add it to conditioning — or accept ~3% as
the floor for this catalogue's truth columns. Improvement goal (m<+3%) NOT achieved; this is an
honest, triangulated negative result.

## 2026-06-25 — Selection/detection RULED OUT as the bias source (`isolate_selection_effect.py`)

Tested the hypothesis that the g=0.05/0.2 *detected* sample is a different selection than the
g=0 sample the flow trained on (detection is a measured-space threshold, shear-dependent through
shape). Binned by true size (controls the covariate shift) x true brightness `r_input_p`, with
undetected rows kept to measure completeness. Decisive result:

```text
sizebin magbin  detf_g0 | g=0.05 detf ratio | g=0.2 detf ratio
0  BRIGHT  0.923 |  0.923  0.881 |  0.923  0.880
0  FAINT   0.352 |  0.353  0.949 |  0.351  0.884
1  BRIGHT  0.915 |  0.916  0.856 |  0.915  0.867
1  FAINT   0.358 |  0.360  0.917 |  0.357  0.908
2  BRIGHT  0.907 |  0.907  0.938 |  0.906  0.920
2  FAINT   0.311 |  0.311  0.941 |  0.308  0.994
overall det_frac: g0=0.502  g0.05=0.503  g0.2=0.502
```

- **Detection is shear-INDEPENDENT**: detf is identical across shears in every (size,mag) cell
  (overall flat to 0.1%). Cells are binned by TRUE size/brightness, so shear only moves the
  shapes inside them and detection doesn't respond. A selection that doesn't change with shear
  cannot bias shear -> **selection is not the source.** (Consistent with p_cat making m worse:
  the true shape-shear selection response is ~0, so the learned P(s=1) term only injects its own
  error; and with the doc's note that selection response is size/flux/magnification-dominated.)
- The forward/data ratio is != 1 (~0.86-0.94) in **every** cell, **including BRIGHT,
  detection-stable, fixed-size cells** (under-predict 8-14%). The gap survives removing selection,
  brightness, and (binned) size -> the residual is **intrinsic to the shape likelihood at matched
  true properties**: at the same catalogue truth, sheared galaxies show a slightly larger shape
  response than g=0 galaxies. Points to insufficient conditioning truth / a rendering-level hidden
  variable, NOT selection and NOT (binnable) covariate shift.
- Caveat: ratio is the first-moment proxy (~0.88 -> first-moment m ~+14%); the flow-MLE extracts
  the full density and lands at +3%. The relative pattern (shear-independence of detf; bright-cell
  under-prediction) is what isolates selection and is robust.

## 2026-06-25 — Forward-model fidelity is ANTI-correlated with low m (key negative result)

Tested whether a richer / size-dependent shape response in the mean head beats meanblind's
+3.0%. Three shape-only flow variants, each with a chained 1M-row baseline recovery (per-model
output dirs to avoid the earlier filename collision):

```text
variant                  s_hat(0.05)  s_hat(0.2)        m       note
meanblind (ref)            0.05137     0.20589     +0.0301    underpowered LINEAR mean head
meanmlp                    0.05292     0.21312     +0.0680    size-dependent MLP mean head
meanmlp_nonblind           0.05300     0.21236     +0.0624    MLP head + shape-seeing residual
meanblind_bigcap           0.05310     0.21267     +0.0638    linear head, 16 flows x 384
```

**All three richer variants REGRESSED to ~+6%** — the plain-shape2d level — none beat meanblind.
This inverts the working hypothesis and is the important finding: **giving the forward model
more capacity to faithfully fit the g=0 response makes m WORSE, converging to the structural
transfer floor (~+6%) that the first-moment forward analysis already predicted.** The meanblind
+3% is therefore NOT a principled optimum; it is a *fortuitous underfit* — the underpowered
linear mean head overshoots the shape response in the helpful direction, partially cancelling
the covariate-shift bias by accident. The moment the model fits g=0 honestly (MLP head, or more
flow capacity), that lucky cancellation disappears and m regresses.

Consequence: **m cannot be reduced by improving forward-model fidelity.** This is strong
model-side confirmation (independent of the first-moment/HGB isolation) that the residual is a
g=0 -> sheared *covariate-shift transfer gap*, not a density-fit deficit. The "enrich the flow"
route is exhausted.

### Two untried legitimate levers now testing (job 14807225, recovery-only, meanblind)

1. **p_cat = p_meas * P(s=1).** The baseline m uses `p_meas` ALONE; the framework's actual
   objective includes the selection term (the selection *response* to shear is a real channel
   the doc emphasises, SBI_shear.md S2). Adding `log P(s=1|S_s(x),n)` with
   `selection_mlp_g0_shearfree_v1` may shift m. Untried on the baseline so far.
2. **Closure on meanblind** (targets drawn from the flow at known s0=0.05/0.2): if the scan
   recovers s0 exactly, ALL of +3% is irreducible transfer error; any residual is fixable
   *estimator* bias (1D projection, secondary-at-intrinsic approx, grid/quad-fit).

If neither helps, the honest position is that +3% is the floor for the pure forward-model route
and sub-percent needs the rendering-level fix (covariate-shift-robust training or the galsim
matched-render experiment, see 06-24 synthesis).

## 2026-06-24 — Drive multiplicative bias m toward sub-percent (model fidelity, not calibration)

Goal (user): improve the raw m of the held-out-shear recovery to sub-percent **legitimately**
— forward-model fidelity only, **no empirical removal** (do not measure m on the sheared sims
and divide it out; that is circular and uses the shear truth). Reasonable quality cuts allowed.

### Meanblind result (was trained 06-24 but never logged)

The shape-blind residual mean flow `measurement_flow_g0_shape2d_meanblind_v1` (job 14798663,
linear mean head *learned* by ML, residual flow blind to `e1/e2_input_p`) landed:

```text
recovery (no cut): s_hat(0.05)=0.05148, s_hat(0.2)=0.20597
formal m = +0.0299 +/- 0.0004   (was +0.059 for plain shape2d affine) -> ~halved
c1 = -0.00082, c2 = -0.00043  -> additive already PASSES Stage-IV (|c|<1e-3)
val NLL 2.08 (vs 1.64 for the non-blind mean flow: blindness costs density fit)
```

So additive is fine; the whole gap is the constant multiplicative m. m is the standard WL
multiplicative bias `g_hat=(1+m)g+c` (compute_mc_bias.py: slope-1 of s_hat vs g over the two
shears), here ~equal to the per-shear fractional error because c~0 and the response is linear.
**m halved but is still ~10x over the 3e-3 target.** Mean head halved it; a *learned* linear
mean does not pin the response because the ML mean != OLS mean for non-Gaussian residuals.

### Diagnosis path

Closure (earlier) already proved the recovery is unbiased *given a faithful model* and S_gamma
is exact vs galsim, so m is purely a model-fidelity gap: the flow's conditional-mean shape
response `M_model` (~0.205-0.21) is below the data response `M_data~0.252` (OLS at g=0).
Pinning `M_model == M_data` targets m at its root. M_data is a pure g=0 quantity, so deriving
the response from it is legitimate self-calibration, NOT the illegitimate sheared-sim m-removal
in `responsivity_bias.py`.

### Added / changed (SBSI)

- `sbs_shear/measurement_model.py`: `ConditionalMeanFlow.set_ols_mean_and_freeze(context_std,
  target_std)` — fit the linear mean head by OLS on the g=0 standardized (context, target) and
  freeze it, so `M_model == M_data` by construction and ML training cannot shrink it. Paired
  with the shape-blind residual flow, the shape->shape response is exactly the data response.
- `scripts/train_measurement_model.py`: `--freeze-mean-ols` (fits+freezes the OLS mean before
  building the optimizer; only trainable params are optimized).
- `jobs/job_diagnose_mmodel.sh`: measure M_model for meanblind / mean / plain shape2d vs M_data.
- `jobs/job_train_measurement_olspin.sh`: train the OLS-pinned shape-only mean_affine flow
  (`measurement_flow_g0_shape2d_olspin_v1`), residual blind to `e1/e2_input_p`.

### Validation

```text
py_compile measurement_model.py train_measurement_model.py -> OK
smoke (CPU): set_ols_mean_and_freeze recovers a planted 0.25 diagonal response
            (0.2496/0.2516, off-diag ~0), mean head frozen, flow trainable, log_prob finite
```

### Jobs (running)

```text
14800852 SBSI_MDIAG       M_model(meanblind/mean/shape2d) vs M_data
14800860 SBSI_MEAS_OLSPIN train OLS-pinned shape-only flow
14800861 SBSI_RECOVER     recovery g=0.05/0.2 (afterok:14800860)
```

### M_model diagnostic — m does NOT track the conditional-mean response (14800897)

Measured M_model (flow finite-diff shape response) vs M_data=0.2524 (OLS, g=0):

```text
plain shape2d (pure flow):     M_model=0.206 (ratio 0.82, UNDER-responds)  recovery m=+0.059
meanblind (learned mean head): M_model=0.274 (ratio 1.09, OVER-responds)   recovery m=+0.030
```

**Key negative result:** recovered m does not track |M_model - M_data|. Both an under- and an
over-responding conditional mean give m>0, and the *overshoot* gives the *smaller* m. So the
flow-MLE bias is set by the full conditional density, not the mean response -- pinning the mean
to M_data is necessary (it is the correct g=0 value) but NOT sufficient for m->0. (The OLS-pin
flow 14800860 is still the cleanest flow test: exact-OLS mean vs meanblind's overshoot.)

### Forward first-moment estimator + root-cause diagnosis (the important part)

Built `scripts/responsivity_estimator_g0.py` (+ job): the framework's weak-shear linearization
with the responsivity derived ONLY from g=0 + the exact analytic S_gamma map (Mobius), inverting
the sheared first moment <chi_par>. NOT responsivity_bias.py (which sets R=<chi_par>/g from the
known shear -- circular). Three g=0 forward response models: `reduced` (linear chi~M*eps),
`distortion` (exact chi=2eps/(1+|eps|^2)), `cubic` (free isotropic E[chi|eps]).

Full-stats m (3M rows/config; <chi_par>/g_data = 0.232):

```text
reduced    R_g0=0.251 -> m(0.05)=-0.060, m(0.2)=-0.062   (misses distortion responsivity)
distortion R_g0=0.247 -> m(0.05)=-0.043, m(0.2)=-0.027   (better, exact shape map)
cubic      R_g0=0.250 -> similar; the gap is NOT a shape nonlinearity
```

So the forward first-moment estimator UNDER-recovers ~3-6% (m<0); the flow-MLE OVER-recovers
~3-6% (m>0). **They bracket zero.** Additive c1,c2 pass throughout.

**Decisive per-object fidelity check:** shear each sheared object's OWN intrinsic shape by its OWN
applied gamma (exact Mobius), predict measured shape via the g=0 conditional mean E[chi|scene
shape], compare to actual. Result (full stats): predicted **over-predicts** the actual measured
shape, and the over-prediction **grows with shear** (~+3% at g=0.05, ~+8% at g=0.2). So the gap
is a genuine forward-model fidelity issue, not the inversion.

**Root cause (physical):** conditioning incompleteness. Shear elongates *round* galaxies, whereas
a g=0 object at the same scene ellipticity is *intrinsically* elongated -- different size/profile,
hence different PSF dilution. The measured-shape response depends on size, not just scene shape.
The catalogue `Re_input_p` is moreover the *pre-shear* size; the rendered (measured) size is
sheared (magnification det A = 1-|g|^2). A naive linear size-interaction over-corrected (full-stats
TBD), so the size term needs the rendered/post-shear size, not pre-shear Re.

### Status vs the goal (interim, later superseded below)

Sub-percent m NOT reached. Best: flow-MLE meanblind m=+0.030; forward distortion m=-0.027..-0.043.

### CORRECTION — the sim shears shape only; there is NO magnification to apply

Checked the renderer: `MultiBand_ImSim/modules/ImSimObject-gen.py` applies the applied shear with
GalSim `galaxy.shear(g1=gamma1, g2=gamma2)` — pure reduced-shear distortion, **flux- and
area-preserving, no `.lens()`/`.magnify()`**. The catalogue `Re`/`r` are intrinsic and unchanged.
Therefore the recovery shearing ONLY the primary ellipticity (holding size/flux/separation fixed)
is the EXACT, self-consistent inverse of the simulation. The earlier "magnification/rendered-size
fix" is WRONG and is retracted (task #6 dropped). The residual m is purely **conditional-density
fidelity**, not shear-application.

### Reliable 1M-row recovery (the 100k runs were sampling-noise-limited, +-0.04 in m)

```text
                 baseline   SNR>20    isolated
meanblind (learned mean) +0.030   +0.121    +0.052
olspin   (frozen M_data) +0.079   +0.167    +0.102
```

- **OLS-pin is WORSE than meanblind** (+0.079 vs +0.030): pinning the conditional mean to the g=0
  data response M_data=0.252 does NOT reduce m. Confirms m is not set by the conditional-mean
  response (the meanblind mean head overshoots to M_model=0.274 yet recovers better). Negative
  result; OLS-pin abandoned.
- **SNR>20 strongly WORSENS m** (both models, ~+0.09). An SNR cut is a selection on a *measured*
  quantity and p_meas has no P(SNR>20|true,shear) factor, so it injects selection bias rather than
  removing it. Lesson: model selection (p_cat), do not cut around it.
- **isolated also worsens m** (blended galaxies have lower m); blending is not the bias driver.
- NOTE: recovery m at 100k rows scatters +-0.04 (weak shape signal); use >=1M rows. Output dirs are
  now per-model (`results/heldout_shear_recovery/<model>/`) to stop concurrent jobs colliding.

### Isolating the source: progressive conditioning (`scripts/isolate_bias_conditioning.py`)

Per-object forward fidelity with a flexible regressor E[chi|feature_set] (HistGradientBoosting,
fit g=0, predict S_gamma(intrinsic)+own features, compare pred/act). Smoke (110k):

```text
feature set     g=0.05 pred/act   g=0.2 pred/act
shape           1.064  (+0.064)   1.051 (+0.051)   over-predicts (the known bias)
+size           0.866  (-0.134)   0.820 (-0.180)   FLIPS to under-predict!
+flux/+sersic   ~0.90  (-0.10)    ~0.86 (-0.14)
```

Adding size doesn't fix it, it OVER-corrects and flips the sign — robust to the regressor
(not a linear-term artifact). Signature of **covariate shift / transfer failure**: at g=0 shape
correlates with size, but shear shifts shape independently of size, so a model exploiting the g=0
shape<->size correlation mis-transfers to the sheared population. Full-stats run (14806969) confirms
at 1.5M/1M rows + measures the g=0 shape-size correlation + a regularization scan (overfitting vs
covariate shift).

### Quantified covariate shift (`scripts/responsivity_size_binned.py`)

```text
corr(|e_scene|, log size) = +0.42  (strong)         per-bin response varies 10x: 0.034..0.327
R_marginal (shape-only slope)      = 0.248  -> m(0.05)=-0.088, m(0.2)=-0.059
R_conditional (<within-size-bin>)  = 0.210  -> m(0.05)=+0.076, m(0.2)=+0.111
data responsivity <chi_par>/g      = 0.232  (BETWEEN the two)
```

So the true response sits BETWEEN the marginal (over-estimates -> m<0) and the intrinsic-size-
conditional (under-estimates -> m>0). The bias is not "missing size conditioning"; it is that
(i) shear shifts shape independently of size while g=0 has them correlated (+0.42), and (ii) the
intrinsic-size-conditional slope itself under-estimates the response (likely the rendered/measured
size grows under shear -> less dilution -> extra response, a feedback the intrinsic-Re binning omits).
The full per-object forward (the FLOW, conditioning on size/flux) is the right tool; its residual
+3% is where this lands. This is the honest isolation of the multiplicative-bias source.

### Full-stats isolation (14806969) — regularization-INDEPENDENT (the decisive test)

```text
corr(|e_scene|, .): logRe=+0.417  sersic=-0.155  mag=-0.007   (size & morphology, NOT flux)
HGB E[chi|featset] pred/act (m_impl), g=0.05 / g=0.2:
                   flexible(63 leaves)     heavy-reg(15 leaves, min_leaf 20k)
  shape            +0.018 / +0.066         +0.021 / +0.066
  +size            -0.179 / -0.182         -0.176 / -0.176
  +flux            -0.132 / -0.131         -0.135 / -0.130
  +sersic          -0.137 / -0.133         -0.137 / -0.133
```

Flexible and heavily-regularized are IDENTICAL -> the +size over-correction is NOT overfitting/
sparse-corner noise; it is a genuine covariate-shift effect. Adding size flips the forward from
+7% over-prediction to -18% under-prediction, robustly. Flux is irrelevant (corr ~0). So the bias
source is firmly: the g=0 shape<->(size,sersic) correlation makes the conditional-mean forward
non-transferable to the sheared population.

### Synthesis (bias source isolated; honest improvement status)

- m is forward-model fidelity error (closure proves the framework is unbiased given a faithful
  density). Ruled out as fixes/causes: shear-application (sim shears shape only, no magnification),
  conditional-mean pinning (OLS-pin WORSE), SNR cut (worse, selection not modeled), blending (not
  the driver), flow architecture (affine==spline), noise/method (closure unbiased, 1M-row stats).
- ISOLATED source: g=0 shape<->(size,sersic) covariate shift. The shape response varies 10x with
  size; shear decorrelates shape from size; the true responsivity (0.232) sits between the marginal
  (0.248) and size-conditioned (0.210) estimates. Simple conditioning over-corrects, regardless of
  regularization.
- BEST estimator remains the flow-MLE meanblind at m=+0.030 (uses the full conditional density, so
  it beats every first-moment forward, which land at -6%/+8%). Sub-percent NOT reached.
- Concrete future paths to sub-percent (each substantial): (a) covariate-shift-robust flow training
  (importance-weight g=0 to cover the sheared (shape,size) product distribution); (b) galsim
  rendering experiment to resolve why g=0 vs sheared galaxies at matched (scene-shape, intrinsic-Re)
  measure ~10% differently (elliptical-PSF x shear-rotation, or rendered-size feedback); (c) a
  selection model in measured space so quality cuts (SNR) can be used without injecting bias.

## 2026-06-23 (afternoon) — Move to implementation on the live blendemu run

Began executing the refined g=0 framework against real data. Key data-state findings
came first and changed the plan; recording them honestly.

### Data-state audit (important)

- **`/project/.../lsst_selec_emu/` is deleted.** Every derived SBSI catalogue the
  earlier models (selection v1–v6, measurement/scene flows) trained on lived there and
  is gone. Those checkpoints are now orphaned from their training data and from the
  refined framework; treated as reference-only.
- **Canonical dataset is the latest blendemu run** `lsst_sims_fs2_25876/`
  (config `blendemu/configs/fs2_lsst_r.yaml`, input `FS2_25876`, built 2026-05-21→29).
  Shear grid is now **{0.0, 0.05, 0.2}** (response 0/0.2, self_response 0/0.05), not the
  old ±0.1/±0.05 5-point grid.
- **The live detection catalogue is `shear_case = 0.05` only and has no `measured_*`
  columns** (scanned the full file). Root cause: the rewritten blendemu pipeline
  (`run_pipeline.py`) builds detection at the sheared self-response case by default
  (`_detection_shear_label` → 0.05) and does not pass `include_measured`. The g=0 and
  g=0.2 renderings exist on disk; they were simply never turned into catalogues.
- The old standalone `blendemu/scripts/build_detection_catalogue.py` was removed in the
  pipeline rewrite, but its engine `blendemu.response.retrieve_detection(shear=…,
  include_measured=True, k=…)` still exists and is what the pipeline calls.

### Decisions (with user)

- Build everything on `lsst_sims_fs2_25876`; retrain fresh at g=0 (old models reference-only).
- Reconstitute the training catalogue via a **thin SBSI-side builder** (keep blendemu
  untouched), nearest-pair **k=2**, `include_measured=True`.
- `p_meas` targets = the SExtractor detection-join observables.

### Added / changed (SBSI)

- `sbs_shear/selection_model.py`: added `SHEARFREE_G0_SELECTION_FEATURES` (14 feats; v6
  primary-frame set minus the applied-shear `gamma_pframe_*` inputs) and
  `SELECTION_FEATURE_SETS`.
- `sbs_shear/measurement_model.py`: added `SHEARFREE_G0_MEASUREMENT_CONDITION_FEATURES`
  (16; g0 selection feats + primary/neighbour redshift) and
  `MEASUREMENT_CONDITION_FEATURE_SETS`.
- `scripts/train_selection_model.py`, `scripts/train_measurement_model.py`: added
  `--feature-set {v6_primary_frame,g0_shearfree}` and `--shear-case` (g=0 filter);
  measurement default catalogue/output repointed off the deleted path.
- `scripts/build_detection_measurement_catalogue.py`: NEW thin builder calling
  `blendemu.response.retrieve_detection` per case for a given shear, with
  `include_measured=True`, k=2, r_max=3; per-batch feather + merge; skips missing cases.
- `sbs_shear/shear_map.py` + `tests/test_shear_map.py`: NEW analytic S_gamma map
  (Mobius reduced-shear `eps'=(eps+g)/(1+conj(g)eps)`, inverse, at-zero Jacobian
  `J=[[1-a,-b],[-b,1+a]]`, A-matrix separation, magnification). Tests pass: inverse
  round-trip, finite-difference Jacobian (atol 1e-5), orientation-average J→identity
  (unit responsivity for eps), magnification.
- `scripts/validate_heldout_shear_recovery.py`: NEW. Marginal-likelihood held-out-shear
  recovery — feeds `S_{s*ghat}(intrinsic)` to the g=0 measurement model along each sheared
  object's own applied direction, scans magnitude `s`, expects the mean-log-prob curve to
  peak at `s=|g_applied|` (and at ~0 for the unsheared half). Frame-agnostic; writes
  `.npz` + `.png`.
- Jobs: `job_build_detection_measurement_catalogue.sh`,
  `job_train_selection_g0_shearfree.sh`, `job_train_measurement_g0_shearfree.sh`,
  `job_validate_heldout_shear_recovery.sh` (chained `afterok` on the measurement model;
  runs g=0.05 sheared+unsheared and g=0.2 sheared).

### Validation / runs

```text
py_compile: selection_model, measurement_model, train_selection, train_measurement,
            build_detection_measurement_catalogue  -> OK
shear_map tests: all pass (sims1)
builder smoke (sims1, g=0 cases 0-1): 1,399,136 rows, 61 cols, 25 measured_* cols,
            detected=0.449, neighbored=0.783, gamma_input==0 exactly,
            measured_* finite for 100% detected / 0% non-detected
```

Catalogue builds (Slurm, partition=cluster, k=2 nearest-pair, include_measured):

```text
14770537 g=0.05 cases 0-99  -> det_meas_g0.05_val.feather  (~18 GiB, 374s) COMPLETED
14770538 g=0.2  cases 0-99  -> det_meas_g0.2_val.feather   (~18 GiB, 367s) COMPLETED
14770536 g=0.0  cases 0-199 -> det_meas_g0.0_train.feather (139.9M rows)   merging
```

Training jobs submitted with `--dependency=afterok:14770536` (auto-start when g=0 lands):

```text
14770590 SBSI_SEL_G0   g0_shearfree selection P(s=1|x,n), shear_case=0.0
14770591 SBSI_MEAS_G0  g0_shearfree measurement flow p_meas(x_hat|x,n), shear_case=0.0
```

### Shear-application scheme (non-obvious; verified from data + code)

`blendemu.catalog.generate_catalog_realization(..., shear_type='constant')` shears
**only the second half** of each case's galaxies, each with magnitude = the case shear
(0.05 or 0.2) at a **random orientation**; the first half is unsheared (g=0). So in a
"g=0.05" catalogue ~half the rows have `|gamma_input|=0.05` (random direction) and ~half
have `gamma_input=0`. Confirmed empirically: case rows are ordered [unsheared half][sheared
half]; sampled batches showed all-zero, all-0.05, or a mix at the boundary with direction
std ~44 deg. **Implication:** ground-truth shear is *per-object* (direction known from
`gamma1/2_input`), not a single value per case. The held-out-shear recovery test is
therefore a *per-object response* test — feed `S_{s·ghat}(intrinsic)` to the g=0 model and
scan the magnitude `s` along each sheared object's own applied direction; the marginal
likelihood should peak at `s = |g_applied|` (0.05 / 0.2), and at `s≈0` for the unsheared
half (control). This is the frame-agnostic form of the doc's "held-out-shear recovery".

### First trained models + recovery result (and a real flaw it caught)

Trained on the g=0 catalogue (4M reservoir-sampled rows):

```text
14770590 selection  g0_shearfree: early-stop ep45, temp 0.994,
         Val logloss=0.2419 brier=0.0711 acc=0.906 AUC=0.962  (cf old v6 AUC 0.938)
14770591 measurement g0_shearfree: early-stop ep55, val NLL ~ -3.35, mean logprob ~3.47
```

Held-out-shear recovery (14770694, 100k rows/run, marginal-likelihood scan):

```text
g=0.05 unsheared (null): s_hat = -0.006  (expect 0)     -> OK, control passes
g=0.05 sheared:          s_hat = +0.006  (expect 0.05)  -> right sign, ~8x too small
g=0.2  sheared:          s_hat = +0.015  (expect 0.2)   -> right sign, ~13x too small
```

**Diagnosis (real modeling flaw, not a code bug):** the measurement flow conditioned on
`e_abs_p` — the primary shape *magnitude* in its own frame, which is rotation-invariant.
The target `measured_e1/e2_image` is orientation-full, and shear acts on the shape
*direction*; with the orientation discarded the model can only respond through the small
magnitude change, hence ~10x under-recovery. The null peaking at 0 and the monotonic
sign/scaling confirm the machinery is correct — it is an information-starved conditioning,
specific to the measurement model (`e_abs_p` is fine for selection, which is
~orientation-independent).

**Fix:** added `g0_oriented` measurement condition set replacing `e_abs_p` with the
oriented sky-basis components `e1_input_p, e2_input_p`. Retrain + re-validate:
`14770906` (measurement, g0_oriented) -> `14770907` (recovery, chained afterok).

**Oriented result (14770906/907):** mean log-prob 3.47 -> 4.12 (orientation makes the
measured-shape density much more predictable). Recovery:

```text
g=0.05 sheared:  s_hat = 0.0335  (67% of truth; was 0.006 / 12%)
g=0.2  sheared:  s_hat = 0.145   (72% of truth; was 0.015 / 7.5%)
g=0.05 unsheared null: s_hat railed to grid edge -0.05  (SEE NULL CAVEAT)
```

The ~6-10x improvement confirms the `e_abs_p` orientation flaw was dominant, and a
67-72% recovery largely vindicates the analytic `S_gamma` map (the likelihood can only
peak near the truth if `S_s(intrinsic)` matches the rendered scene).

**Null-control caveat (design flaw, found via review):** the unsheared run is degenerate
as implemented. Unsheared objects have `gamma_input=0` -> no shear direction
(`ghat=0`) -> `S_s(intrinsic)` is independent of `s` -> the likelihood is flat and the
argmax reads noise/grid-edge. The earlier "-0.006" was noise off a flat curve, not a
passing control. TODO: impose a fixed fiducial direction on the unsheared sample so the
null is a real `s_hat ~ 0` test.

**Test-sample quality cut (SNR>20) — noise ruled out (14771220):** SExtractor SNR =
flux_auto/fluxerr_auto. Cut keeps the brightest 28.5% of detected sheared objects (median
SNR ~13). Recovery essentially unchanged: g=0.05 0.0335->0.0345 (67->69%), g=0.2
0.1447->0.1489 (72->74%). The high-SNR likelihood is much sharper (stat error collapses to
+-0.001), so the shortfall is now ~17 sigma. **The ~26-31% under-recovery is a sharp,
multiplicative (~0.74x), S/N-robust systematic, not measurement noise.** A clean scale
factor identical at 0.05 and 0.2 most implicates an S_gamma-vs-simulator scale/convention
mismatch or model under-response (flow shrinkage / sky->image frame washout); less likely
pure selection. Next decisive test: a model closure test (generate x_hat from the model at
a known shear, recover it) to separate estimator/model bias from physical (S_gamma/PSF/
selection) bias.

**Closure test — recovery method is UNBIASED (14772165):** generate x_hat from the trained
flow at a known shear s0 (conditioning S_{s0}(intrinsic)), then recover. Result:
s0=0.05 -> 0.0496, s0=0.2 -> 0.1996. The estimator (1-D scan along the applied direction,
S_gamma-on-features, quadratic peak, population-mean MLE) returns the truth exactly. So the
~0.74x on REAL data is NOT a method/code artifact -- it is a genuine model-vs-data mismatch
(the model's conditional best matches the real sheared data at 0.74x the true shear).
Remaining suspects narrowed to: (a) flow over/under-responds vs the data
(M_model vs M_data, being measured), (b) S_gamma Jacobian scale vs the simulator,
(c) selection. Noise and estimator bias are ruled out.

**Shape-response diagnostic — the limiter is FLOW FIT QUALITY (14772189):**
`diagnose_shape_response.py` measures M = d(measured e)/d(intrinsic e) on the same g=0
sample, two ways:

```text
M_data  (lstsq, 200k):  diag 0.252, off-diag ~0      (true PSF-diluted response)
M_model (flow FD):      diag 0.199, off-diag -0.03    (learned response)
M_model / M_data = 0.789
```

The flow UNDER-responds to the conditioning shape by ~21% (regression-to-mean) and adds a
small spurious off-diagonal. This 0.79 matches the recovery shortfall (0.67-0.74). With
closure passing (method unbiased), noise ruled out, and this direct M measurement, the
dominant limiter is the affine-coupling flow's fit quality, NOT noise/method/S_gamma/
selection. Action: train a stronger measurement flow (more capacity/flows/data/epochs;
RQ-NSF when available) and re-measure M_model + recovery. (delta_et/gamma ~0.3-0.5 from
blendemu and M_data~0.25 are consistent PSF-dilution scales.)

**Selection factor ruled out + null fixed (14772305):** (a) adding log P(s=1|x,n) to the
likelihood (p_cat) left g=0.05 recovery at 0.0335 -> selection response is negligible here,
NOT a cause. (b) the fixed-fiducial null (direction-less objects assigned ghat=(1,0)) now
recovers s_hat=-0.0002 ~ 0 -> the null is a valid control and passes. Recovery tool now
supports --selection-model (p_cat), --closure-shear, --snr-min/--mag-max, fixed null.

**Elimination summary:** noise (SNR cut) NO, estimator/method (closure) NO, selection
(p_cat) NO, null PASSES. Sole identified cause = measurement-flow under-fit
(M_model/M_data=0.79). Fix in progress: stronger flow (512/16/4, 8M rows, 150 ep) ->
chained M-diagnostic + recovery.

**BREAKTHROUGH — shape-only likelihood fixes the recovery (14781412/413):** trained a
2-target flow on ONLY (measured_e1_image, measured_e2_image) vs the joint 6-target flow.

```text
joint 6-target:  g=0.05 -> 0.0304 (61%),  g=0.2 -> 0.134 (67%)
shape-only 2D:   g=0.05 -> 0.0533+-0.0018 (107%),  g=0.2 -> 0.2122+-0.0019 (106%)
```

M_model/M_data is the SAME (~0.81) for both, so the shortfall was NOT flow shrinkage --
it was the non-shape targets (flux/radius/size), which are ~invariant under shear but
carry spurious shape-conditioning correlations learned at g=0, dragging the joint MLE
toward s=0. Modeling the shape channel alone (physically the shear-carrying observable)
recovers to ~106% at both shears. Recovery went from ~35% under to ~6% over. New best
model: SBSI/models/measurement_flow_g0_shape2d_v1.pt. Cancelled the slow 6D affine-long
run (14781294) as superseded.

Remaining ~6% over-recovery: small, constant multiplicative at both shears (~1.06).

**S_gamma confirmed EXACT (galsim check):** apply_shear_to_ellipticity matches
galsim.Shear reduced-shear composition to machine precision (max |diff| = 0.0) over all
test cases. Intrinsic shape (FS2 README) and applied g1/g2 are both reduced-shear
convention, composed by exactly this Mobius. So the residual ~6% is NOT a mapping/
convention error -- it is the flow's conditional fit (testing RQ-NSF spline) or measured-
shape responsivity nonlinearity. This closes the earlier S_gamma-vs-simulator concern.
RQ-NSF spline flow added (sbs_shear/spline_flow.py, invertibility-tested) + --flow-type.

**Selection-aware form confirmed:** shape-only p_meas x P(s=1|x,n) recovers 0.0534/0.2127
(=107%, identical to shape-only alone) -> the selection factor is ~flat in shear and does
not degrade the shape recovery. This is the clean factorization the user wants:
p_cat = p_meas(shape|x,n,s=1) . P(s=1|x,n), with flux/size living in the selection
classifier (which conditions on truth size/flux), NOT conditioned-on in the shape density.

**Spline-flow bug (fixed):** SplineCoupling._apply collided with nn.Module._apply (called
by .to(device)); renamed to _transform. The CPU smoke missed it (never called .to). GPU
job caught it.

**Spline result (14782820/821):** shape-only RQ-NSF spline gives M_model=0.208 (ratio
0.826) and recovery 0.0530/0.2118 (106%) -- essentially IDENTICAL to the shape-only affine
(M_model 0.204, recovery 0.0533/0.2122). The spline trains far more stably (val NLL flat
~1.507 vs affine's +-1.5 bouncing) but does NOT change recovery. So the residual ~6%
over-recovery is ARCHITECTURE-INDEPENDENT -- not flow expressivity. Both flows shrink
M_model to ~0.21 (vs M_data 0.25), yet recover 106%, so it is also not simple mean-
shrinkage. Remaining candidates: (a) blend approximation (recovery shears only the primary,
keeps secondary at intrinsic) -- testing isolated vs blended (14787314); (b) a measured-
shape responsivity/convention factor (measured_e=(A-B)/(A+B) from SExtractor RMS sizes vs
the reduced-shear convention) -- a small constant multiplicative is exactly what shear
pipelines calibrate (the sim's response/self_response catalogues exist for this).
Production shape model = affine (spline no better, slower).

**Blend diagnostic (14788976) -> residual is a constant responsivity, fully characterized:**
g=0.05 shape-only recovery by neighbour status: all 0.0535, isolated 0.0528, blended 0.0530
(all ~106%). Isolated and blended over-recover IDENTICALLY, so the ~6% is NOT the
primary-only-shear blend approximation. The ~6% over-recovery is now confirmed uniform
across: shear magnitude (0.05==0.2), S/N (SNR>20), flow architecture (affine==spline), and
blend status (isolated==blended). => it is a single constant multiplicative measured-shape
responsivity factor (m ~ +0.06), i.e. the SExtractor RMS-size ellipticity (A-B)/(A+B) vs
the reduced-shear convention. This is exactly the multiplicative-bias calibration every
shear pipeline applies, measurable from the sim's response/self_response catalogues. It is
a calibration step, not a flaw.

## 2026-06-23 (late) — Formal m/c calibration bias vs Stage-IV

Goal raised to LSST-era (Stage-IV/"Stage-VI") shear calibration: |m|<~3e-3, |c|<~1e-3
after quality cuts. Measured the formal bias g_hat=(1+m)g+c on the shape-only model
(300k objects/config; scripts/compute_mc_bias.py, errors from weighted-polyfit cov):

```text
s_hat(g=0.05) = 0.05331 +/- 0.00003
s_hat(g=0.20) = 0.21217 +/- 0.00005
m  = +0.0591 +/- 0.0004   -> FAIL (20x over 3e-3); constant (same at 0.05 & 0.2 => linear)
c1 = -0.00082 +/- 0.00002 -> PASS (<1e-3)  [g=0 recovery along fixed axis ang0]
c2 = -0.00043 +/- 0.00002 -> PASS (<1e-3)  [ang45]
```

Additive ALREADY meets Stage-IV; the multiplicative m~0.06 is the whole gap. It is a clean
constant driven by the flow's conditional-mean under-fit (M_model 0.21 < M_data 0.25 =>
over-recovery). Path: (1) quality cuts (m vs S/N, mag -- job 14795411), (2) responsivity
calibration of the constant m (validate residual on held-out shears), (3) deeper fix =
explicit-conditional-mean measurement model so M_model->M_data and raw m->0.
Added scripts/compute_mc_bias.py (m,c fit + Stage-IV verdict) and --fiducial-angle-deg
(measures c1 at 0deg, c2 at 45deg).

**Quality cuts do NOT reduce raw m (14795411):** m(no cut)=0.0591, SNR>20=0.0616,
SNR>40=0.0621, mag<23.5=0.0599. S/N cuts slightly WORSEN it. So the bias is not low-S/N
shrinkage; it is a fundamental measured-shape response under-fit, uniform across the
population. => the fix is the model, not cuts.

**Explicit-conditional-mean model (implemented):** ConditionalMeanFlow in measurement_model.py
(flow_type mean_affine/mean_spline): p(x|c) = p_resid(x - mu(c) | c) with a direct linear
mean head mu(c). Pins the conditional-mean response (which OLS shows is M_data=0.25, not the
flow's shrunk 0.21) so the flow models only residual scatter -> M_model should -> M_data and
raw m -> 0 without calibration. Training shape-only mean_affine (14795522).

**Model-free measured-shape responsivity (the cause):** R = d<e_parallel>/dg measured
directly from data (frame is sky/image-aligned, off-diag~0):

```text
R(0.05)=0.2327, R(0.2)=0.2337  -> LINEAR to 0.4%, <e1>_unsheared~1e-4 (no additive)
R = M_data x <1-e^2> = 0.252 x 0.92 = 0.233  (exact: the eps-convention Mobius responsivity)
```

So the DATA responsivity is clean and linear; the flow-MLE's m=0.059 is an ESTIMATOR
artifact (mild-nonlinear: per-point m 0.066->0.061), NOT a data/convention/nonlinearity
problem. The mean-head model (ConditionalMeanFlow) did NOT reduce it (M_model still 0.205,
recovery 107%) -- the conditional residual flow re-absorbs the mean (unidentified), and the
net response is robustly ~0.205 across affine/spline/mean architectures.

**IMPORTANT (correction):** empirically measuring m on the sheared sims and dividing it out
(responsivity_bias.py / image-sim calibration) is NOT a legitimate result -- it uses the
known-shear truth, is circular, and defeats the forward-model self-calibration goal. The
legitimate standard: the response is DERIVED from the g=0 model via S_gamma and m->0 must be
VALIDATED on held-out shears the model never calibrated to. Closure already showed the
estimator is unbiased GIVEN a correct model, so m=0.059 is a pure MODEL-FIDELITY gap:
the flow underfits its own g=0 conditional mean (M_model 0.205 vs M_data 0.252, where 0.252
is correct -- it predicts the measured R=0.233=M_data<1-e^2>). A model with M_model=M_data
gives m->0 by construction (exact for linear-Gaussian). So the task is MODEL FIDELITY, not
calibration.

Fix (legitimate): shape-blind residual mean flow -- ConditionalMeanFlow now supports
flow_drop_indices so the residual flow is BLIND to the shape features (e1/e2_input_p);
the shape->shape response is forced into the explicit linear mean head (cannot be shrunk/
re-absorbed). --flow-blind-features in the trainer. Training shape-only mean_affine blind
to e1/e2_input_p (14798663); if M_model->0.252 and recovery->100% on held-out shears, that
is a legitimate self-calibrated m->0.

## STATUS SUMMARY (recovery arc)

Held-out-shear recovery: 61-67% (naive joint flow) -> 106% (shape-only), fully root-caused.
Factorization: p_cat = p_meas(shape | x,n, s=1) . P(s=1 | x,n). Selection-aware p_cat
confirmed (107%, selection factor flat in shear). S_gamma exact vs galsim. Closure unbiased.
Null passes. Ruled out: noise, estimator/method, selection, flow architecture, blending.
Residual: constant +6% responsivity (calibratable). Models: selection_mlp_g0_shearfree_v1
(AUC 0.962), measurement_flow_g0_shape2d_v1 (shape-only, production). Figure:
results/heldout_shear_recovery/recovery_summary.png.

**(superseded) earlier shortfall candidates:**
0. measurement flow under-fits the shape response (M_model/M_data=0.79). [not the main cause]
1. Omitted selection factor: the recovery uses only `p_meas` on a `detected==True` sample;
   the full density is `p_cat = p_meas * P(s=1|x,n)`. The selection response (shear changes
   which objects are detected) is not in the likelihood. Selection model is trained
   (AUC 0.962) and ready to fold in.
2. Residual `S_gamma`-vs-simulator mismatch / PSF dilution / weak-shear linearization
   (g=0.2 is strongly nonlinear). A model-free galsim check of the `S_gamma` convention is
   the clean isolator.
3. Only the primary shape is sheared in the scan; the secondary's shape is kept at its
   intrinsic (rot0) value (not re-sheared by its own gamma_input_s) -- a small
   approximation. Blending IS included: we condition on the KNOWN true neighbour
   properties (size/flux/distance/sersic/redshift/shape, gated by `neighbored`), i.e. we
   do not yet marginalize over a neighbour/population prior (real data will need that).

### Known limitations / next

- All on the SExtractor `_rot0` realization (`use_pos: detect`); single rotation.
- Validation catalogues capped at 100 cases each; ~half of those rows are the sheared set.
- g=0.2 is strongly nonlinear; the single-magnitude scan along the applied direction is a
  first recovery probe, not the full 2D marginal-likelihood inference.
- Next: held-out-shear recovery (recover g on 0.05/0.2 via S_gamma response), model-free
  measured-shape response diagnostic, then hyperparameter/model-choice sweeps.

## 2026-06-23

### Framework Refinement (round 2): Responsivity Bookkeeping

Follow-up to the shear-free reframing below, after a longer discussion clarifying where
the responsivity actually lives. Doc-only edits to `SBSI/SBI_shear.md` §2.

- **$S_\gamma$ scoped down.** Only its *derivative at $\gamma=0$* enters the science (it is
  the responsivity, computed from the prior, never applied to data). The full nonlinear
  map is used only for exact finite-shear inference and for validation/analytic shear
  injection (§6). Added a "Where $S_\gamma$ is actually used" note.
- **Estimator rebalanced.** Lead with the marginal likelihood as primary, with the shear
  **response internal** (the responsivity is the likelihood curvature $\mathcal{L}''(0)$,
  not an external factor). The $\hat\gamma=\langle e_\text{true}\rangle/R$ point estimate
  is its weak-shear linearization; clarified that *naively averaging posterior-mean shapes
  undershoots* (shrinkage against the shear-free prior) and the full inference corrects
  this self-consistently. Once corrected, the estimator has unit response by construction
  — no residual responsivity.
- **Two-responsivity distinction added** (verified against literature). (i) Measurement /
  classical responsivity $\mathcal{R}=2(1-e_\text{rms}^2)$ (Bernstein & Jarvis 2002):
  shape dispersion + ellipticity convention + PSF dilution + selection; prior-free;
  survives perfect measurement; vanishes for reduced-shear $\varepsilon$; this is what
  metacal/metadetect measure from data. (ii) Prior shrinkage: the residual $R<1$ from
  inverting against a shear-free prior; this one **is** the prior — exists *because shear
  is not in the prior* (sheared prior gives $R=1$), and $\to1$ as S/N$\to\infty$. In our
  pipeline only the shrinkage piece remains (we use $\varepsilon$ and absorb PSF dilution
  in the forward model), so $R$ is prior-dominated for us — a bookkeeping outcome, not a
  universal fact.
- Upgraded the citation footnote: B&J 2002, Sheldon & Huff 2017, Sheldon et al. 2020/2023,
  BFD 2014, lensfit verified; MacCrann blending + 2024–25 LSST-DESC updates still
  TODO-verify.

Validation: doc-only; no code touched. Literature checked via web search
(metacalibration ApJ; metadetection arXiv:2303.03947; adaptive-moments responsivity A&A;
BFD arXiv:1403.7669).

### Framework Refinement: Shear-Free Forward Model + Prior-Aware Inference

Refined the mathematical framework in `SBSI/SBI_shear.md` (doc only; no code, feature,
or model changes yet). The central change is removing true shear $\gamma$ as a
conditioning variable.

Reasoning:

- Shear is applied to the *true scene before rendering* (Möbius on intrinsic shape,
  A-matrix shear of separations, flux/size magnification); PSF and noise follow and are
  shear-independent. So at fixed post-shear true scene, $\hat{x} \perp \gamma$
  (sufficiency). The $g=0$ realization already spans the full intrinsic-property range,
  so the forward map carries all responsivity information.
- Train $p_\text{meas}(\hat{x}\mid x,n,s{=}1)$ and $P(s{=}1\mid x,n)$ at $g=0$ only.
  Shear is *inferred*, entering one place: the $S_{-\gamma}$ shift of the true-property
  prior in a BFD-family marginal likelihood. Per-object inversion
  $p(x\mid\hat{x},n)$ gives $\hat\gamma=\langle e_\text{true}\rangle/R$ as the
  weak-shear limit.
- Owned the central caveat: this is prior-dependent (BFD-family), unlike
  metacalibration/metadetection which self-calibrate the responsivity from image shear.
  But single-$\gamma$ image self-calibration cannot resolve redshift-dependent blending
  response (MacCrann+ 2021/22), which is a forward-model quantity — so prior dependence
  is the price of admission for the project's redshift-aware-blending differentiator.

Changes to `SBI_shear.md`:

- §1 motivation: density conditioned on $(x,n)$ only; two outputs (inference + response).
- §2: new "Shear as a distortion of the true scene ($S_\gamma$)" subsection with a
  convention/scope check; sufficiency + $g$-free factorization; shear-free trained
  components; marginal-likelihood inference via prior shift; per-object inverse posterior
  and linearized estimator; differentiable response route as prior-light cross-check;
  corrected the "$\hat e$ estimates $\gamma$" reading via the
  $\partial\langle\hat e\rangle/\partial\gamma = (\partial\langle\hat e\rangle/\partial e')(\partial e'/\partial\gamma)$
  decomposition; new "Prior dependence" subsection (two philosophies + mitigations menu).
- §3 dimensionality: shear removed as a conditioning axis (12D true properties).
- §4: measurement-model note flags the legacy separate-$(e,\gamma)$ features for
  migration to $g=0$ conditioning.
- §6: added held-out-shear recovery (train $g=0$, recover $\hat\gamma$ on $g=0.05/0.1$)
  as the primary end-to-end test, with finite-difference-vs-autograd as the component
  check.
- §9: added a prior-dependence axis to the comparison table.

Known follow-ups (not done here):

- Pin the exact $S_\gamma$ separation/magnification handling and a per-column
  intrinsic-vs-post-shear ledger against MultiBand_ImSim before writing inference code.
- Confirm a $g=0$ LSST realization carries truth shape/size/redshift + neighbour
  annotations + SExtractor measured columns in one place.
- Verify any 2024–25 LSST-DESC shear-pipeline citations before submission.

Validation:

```text
doc-only change; no code touched
grep confirms the only remaining gamma-conditioned p_cat is the sufficiency identity LHS
```

## 2026-05-05

### Scene Training Resubmitted As One Bounded Radial Job

Canceled the two full-catalogue scene measurement jobs because they were still
in the catalogue-loading/grouping phase after about 40 minutes and had not
reached the first `100`-record-batch progress print.

Canceled jobs:

- `13913450` full geometry
- `13913451` radial geometry

Changed:

- Updated `SBSI/scripts/train_scene_measurement_model.py`.
  - Added `--stop-after-scenes`.
  - When set, the loader stops after the requested number of complete scene
    groups instead of scanning the full pair-annotated catalogue for a
    full-catalogue reservoir sample.
- Updated `SBSI/jobs/job_train_scene_measurement_model.sh`.
  - Added `MAX_SCENES` and `STOP_AFTER_SCENES` environment controls.

Validation:

```text
python -m py_compile scripts/train_scene_measurement_model.py
bash -n jobs/job_train_scene_measurement_model.sh
tiny CPU smoke:
  geometry_mode=radial
  max_scenes=128
  stop_after_scenes=256
  load/group/sample time=1.1s
  train_nll=8.483547, val_nll=10.500922
```

Submitted one replacement job:

```text
sbatch --export=ALL,GEOMETRY_MODE=radial,MAX_SCENES=1000000,STOP_AFTER_SCENES=1000000 /home/z/Zekang.Zhang/SBSI/jobs/job_train_scene_measurement_model.sh
```

- Slurm job id: `13913632`
- Initial status: `RUNNING` on `kng-cl-nv01`
- Output target:
  `SBSI/models/scene_measurement_flow_detected_v1_radial.pt`
- Initial log check confirmed:
  - `Using device: cuda`
  - `Geometry mode: radial`
  - `Stop after complete scene groups: 1,000,000`
  - no immediate stderr output

Known limitation:

- This bounded run is not a uniform full-catalogue reservoir sample. It is a
  practical first radial pilot to avoid spending most of the wall time in
  Python-side grouping.

### All-Neighbour Catalogue Verified And Scene Training Submitted

Verified the completed all-neighbour detection-measurement catalogue and
submitted scene-conditioned measurement training jobs.

Catalogue build result:

- Slurm job id `13913352` completed successfully:
  - state `COMPLETED`
  - elapsed `00:07:07`
  - exit code `0:0`
- Output catalogue:
  `/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather`
- Combined catalogue size: `28.67 GiB`
- Catalogue shape from build log: `138,569,015` rows, `62` columns
- File record batches from PyArrow inspection: `2,115`

Lightweight sanity check:

```text
sample read: first 5 record batches, 327,680 rows
detected_rate=0.4374
neighbored_rate=1.0000
distance range: 0.002255..2.999996 arcsec
grouped neighbour-count quantiles:
  0%=1, 25%=1, 50%=1, 75%=2, 90%=3, 99%=4, max=7
finite measured flux/size/axis columns among detected rows: 1.0
```

Submitted scene measurement jobs:

- Full geometry:
  - job id `13913450`
  - output `SBSI/models/scene_measurement_flow_detected_v1_full.pt`
  - initial status `RUNNING` on `kng-cl-nv01`
- Radial geometry:
  - job id `13913451`
  - output `SBSI/models/scene_measurement_flow_detected_v1_radial.pt`
  - initial status `RUNNING` on `kng-cl-nv01`

Initial log check:

- Both jobs selected `cuda`, opened the all-neighbour catalogue, and reported
  the expected geometry mode and input column counts.
- No immediate stderr output.

### All-Neighbour Catalogue Build Submitted

Submitted the all-neighbour detection-measurement catalogue build through
blendemu.

Command:

```text
sbatch /home/z/Zekang.Zhang/blendemu/jobs/job_sbsi_detection_measurement_catalogue_all_neighbors_skycos.sh
```

Important details:

- Slurm job id: `13913352`
- Initial status: `PENDING (Priority)`
- Output catalogue:
  `/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather`
- Build settings from the job script:
  - `--include-measured`
  - `--r-max 3`
  - `--k 32`
  - `--batch-size 1`
  - `--n-jobs 4`

Next step:

- After the Slurm job finishes, inspect the catalogue schema and neighbour-count
  distribution before launching the scene-conditioned measurement model.

### Scene Geometry Modes

Added an explicit radial-only ablation for scene-conditioned measurement
models.

Changed:

- Updated `SBSI/sbs_shear/preprocessing.py`.
  - Adds `e_abs_s`, the scalar neighbour ellipticity amplitude, during
    preprocessing.
  - Allows `raw_columns_for_selection_features(["e_abs_s"])` to request the
    secondary shape columns.
- Updated `SBSI/sbs_shear/scene_model.py`.
  - Adds `RADIAL_SCENE_NEIGHBOR_FEATURES`.
  - Adds `SCENE_GEOMETRY_NEIGHBOR_FEATURES` with `full` and `radial` modes.
- Updated `SBSI/scripts/train_scene_measurement_model.py`.
  - Adds `--geometry-mode full|radial`.
  - Keeps `--neighbor-features ...` as an escape hatch, recorded as
    `geometry_mode="custom"` in metadata.
- Updated `SBSI/jobs/job_train_scene_measurement_model.sh`.
  - Adds `GEOMETRY_MODE="${GEOMETRY_MODE:-full}"`.
  - Default output becomes
    `SBSI/models/scene_measurement_flow_detected_v1_${GEOMETRY_MODE}.pt`.
- Updated `SBSI/tests/test_scene_model.py`.
- Updated `SBSI/SBI_shear.md` and `SBSI/AGENTS.md` with the full-vs-radial
  ablation convention.

Validation:

```text
python -m py_compile sbs_shear/*.py scripts/*.py tests/*.py
bash -n jobs/job_train_scene_measurement_model.sh
manual scene-model test function run in conda env `sims1`: passed
tiny radial CPU smoke train in `sims1` against existing nearest-neighbour catalogue:
  geometry_mode=radial, max_read_batches=1, max_scenes=256, epochs=1
  neighbour features: distance_scaled, Re_input_s_scaled, r_input_s_scaled,
    flux_ratio, sersic_n_input_s, e_abs_s, redshift_input_s
  train_nll=8.496947, val_nll=9.231797
  saved and reloaded /tmp/sbsi_scene_measurement_radial_smoke.pt
```

Known limitation:

- The radial smoke still uses the existing `sbs_skycos` catalogue with `k=2`;
  it validates plumbing, not the all-neighbour radial science case.

### Scene-Conditioned All-Neighbour Measurement Baseline

Added the first DeepSets-conditioned measurement likelihood path for the
one-primary-scene / variable-neighbour-set catalogue design.

Changed:

- Added `SBSI/sbs_shear/scene_model.py`.
  - Defines default primary-scene features, neighbour-set features, and
    `(case, shear_case, input_index)` grouping keys.
  - Adds `SetFeatureStandardizer`, `DeepSetsConditioner`, and
    `SetConditionedMeasurementFlow`.
  - Adds save/load helpers and `load_scene_measurement_model`.
- Added `SBSI/scripts/train_scene_measurement_model.py`.
  - Streams a pair-annotated detection-measurement catalogue.
  - Groups rows by primary scene.
  - Builds a padded neighbour tensor plus mask from all neighbours inside the
    requested aperture.
  - Adds explicit scene summaries: neighbour count, nearest scaled distance,
    brightest-neighbour flux ratio, and total neighbour-to-primary flux ratio.
  - Trains the same normalized selected-object measured-property likelihood,
    but with DeepSets scene conditioning.
- Added `SBSI/jobs/job_train_scene_measurement_model.sh`.
  - Defaults to
    `/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather`.
  - `CATALOGUE=...` and `OUTPUT=...` can override paths at submission time.
- Added `blendemu/jobs/job_sbsi_detection_measurement_catalogue_all_neighbors_skycos.sh`
  to build that source catalogue with `--r-max 3`, `--k 32`, and
  `--include-measured`.
- Added `SBSI/tests/test_scene_model.py`.
- Exported `load_scene_measurement_model` from `SBSI/sbs_shear/__init__.py`.
- Updated `SBSI/SBI_shear.md` and `SBSI/AGENTS.md` with the scene-level
  neighbour-conditioning convention.

Validation:

```text
python -m py_compile sbs_shear/*.py scripts/*.py tests/*.py
bash -n jobs/job_train_scene_measurement_model.sh
bash -n /home/z/Zekang.Zhang/blendemu/jobs/job_sbsi_detection_measurement_catalogue_all_neighbors_skycos.sh
manual scene-model test function run in conda env `sims1`: passed
tiny CPU smoke train in `sims1` against the existing nearest-neighbour catalogue:
  max_read_batches=1, max_scenes=512, max_neighbors=4, epochs=1
  scenes seen before sampling=26,994, scenes used=512
  neighbours per scene: mean=1.000, max=1, zero_frac=0.000
  train_nll=8.495508, val_nll=8.129374
  saved and reloaded /tmp/sbsi_scene_measurement_smoke.pt
```

Known limitations:

- The smoke run uses the existing `sbs_skycos` catalogue, which was built with
  detection `k=2`, so it exercises grouping but remains effectively a nearest
  neighbour catalogue.
- The scene-conditioned job needs an all-neighbour detection-measurement
  catalogue built with sufficiently large blendemu `k` for the chosen aperture.
- The grouped loader still iterates group keys in Python; reservoir sampling now
  avoids materializing every scene, but full-catalogue grouping should stay on
  Slurm.

Next recommended steps:

- Build `/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather`
  from blendemu with `--include-measured`, `--r-max 3`, and `--k` large enough
  for the aperture, for example `--k 32`.
- Submit `SBSI/jobs/job_train_scene_measurement_model.sh` on that catalogue and
  compare held-out NLL against the tabular nearest-neighbour measurement flow.

### SBSI Measurement Likelihood Workflow

Added the first selected-object measurement-density workflow for
`p_meas(xhat | truth, neighbour, shear, s=1)` while leaving the current
selection classifier unchanged.

Changed:

- Added `SBSI/sbs_shear/measurement_model.py`.
  - Defines default measurement conditioning features as the current
    primary-frame selection inputs plus primary redshift and gated neighbour
    redshift.
  - Defines default continuous measured targets: log SExtractor flux, log flux
    radius, log image major/minor axes, and spin-2 image-shape components.
  - Implements a pure PyTorch conditional affine coupling flow, target
    standardization, save/load helpers, log-probability evaluation, sampling,
    and a Monte Carlo mean-gradient helper.
- Added `SBSI/scripts/train_measurement_model.py`.
  - Streams the measured catalogue with PyArrow.
  - Applies current source cuts, conditions on `detected=True`, drops rows with
    non-finite measured targets, trains the conditional density by NLL, and
    writes `SBSI/models/measurement_flow_detected_v1.pt`.
- Added `SBSI/jobs/job_train_measurement_model.sh`.
- Added `SBSI/tests/test_measurement_model.py`.
- Updated `SBSI/sbs_shear/preprocessing.py` so gated neighbour redshift can be
  used as a condition feature.
- Exported `load_measurement_model` from `SBSI/sbs_shear/__init__.py`.
- Updated `SBSI/SBI_shear.md` with the current measurement-model implementation
  note.
- Updated `SBSI/AGENTS.md`, Slurm jobs, and inspection notebooks to use the
  renamed `SBSI` directory instead of the old `SBS` path.

Validation:

```text
python -m py_compile sbs_shear/*.py scripts/*.py
bash -n jobs/job_train_measurement_model.sh jobs/job_train_selection_mlp.sh jobs/job_validate_far_neighbor_invariance.sh jobs/job_evaluate_selection_blends.sh jobs/job_study_selection_gradients.sh jobs/job_study_selection_feature_importance.sh
jq empty notebooks/inspect_detection_catalogue.ipynb notebooks/inspect_selection_model.ipynb
manual test function run in conda env `sims1`: coordinates tests and measurement-model tests passed
tiny CPU smoke train in `sims1`:
  max_read_batches=1, max_rows=512, epochs=1
  source-cut rows=58,920, selected finite rows=26,994
  train_nll=8.501262, val_nll=8.688078
  saved and reloaded /tmp/sbsi_measurement_smoke.pt
```

Known limitations:

- This is a runnable affine-coupling flow baseline, not the preferred RQ-NSF
  backend from the research plan. The active `sims1` environment has PyTorch
  and PyArrow but does not currently provide `zuko` or `nflows`.
- Full-catalogue training and validation should be run through Slurm using
  `SBSI/jobs/job_train_measurement_model.sh`.
- Measurement response gradients from the flow mean are Monte Carlo estimates;
  they need finite-difference and matched-shear validation before science use.

Next recommended steps:

- Submit the measurement-flow Slurm job and inspect held-out NLL plus generated
  target distributions by truth-property bins.
- Add a dedicated measurement-response diagnostic comparing autograd
  `dE[xhat]/dgamma` against finite differences across matched shear cases.

## 2026-05-03

### Expanded Notebook Gradient Profiles

Expanded `SBS/notebooks/inspect_selection_model.ipynb` Section 12 from a
distance-only gradient check into a broader property-profile diagnostic.

Changed:

- Renamed Section 12 to `Core Gradient Profiles By Important Properties`.
- Kept the original neighbour-distance plot with the separate no-neighbour
  reference marker.
- Added pair-only binned gradient profiles for the most informative properties:
  - primary `r_input_p`
  - primary `Re_input_p`
  - primary `e_abs_p`
  - neighbour `distance`
  - `flux_ratio`
  - secondary `r_input_s`
  - secondary `Re_input_s`
  - pair angle relative to the primary major axis
- Extended the Section 12 work table to retain useful context columns alongside
  the autograd gradients and predicted selection probability.
- Added summary tables:
  - binned means and standard errors for each property,
  - per-property overview showing the range and maximum absolute mean gradient
    for each response component.
- Revised the Section 12 plotting layout:
  - one figure contains the distance panel and all important-property panels,
  - each panel overlays the four shear-response components as four curves,
  - axis labels now explicitly report
    `mean selection response <dP(s=1)/dgamma>`,
  - curve labels distinguish
    `dP/dgamma_parallel,p`, `dP/dgamma_cross,p`,
    `dP/dgamma_parallel,s`, and `dP/dgamma_cross,s`.

Interpretation note:

- Expanded property profiles use true-pair rows only. The no-neighbour
  population remains in the distance sanity plot, but is excluded from the
  property scans so secondary-gradient curves are not diluted by rows where the
  secondary response is physically gated to zero.

Validation:

```text
notebook code cells parse OK; outputs=0; execution_counts=0
tiny CPU smoke in conda env `sims1`:
  loaded v6 checkpoint,
  read a 2-batch capped-catalogue sample,
  ran expanded Section 12 with reduced gradient rows,
  ran Section 13 e_abs split downstream from the expanded work table.
additional plotting smoke:
  verified the single-figure four-component panel layout executes.
```

Follow-up fix:

- Fixed a Section 12 pandas `.query()` expression that could raise
  `ValueError: data type must provide an itemsize` when selecting a property
  panel. Replaced the query string with an explicit boolean mask.
- Re-ran the reduced Section 12 smoke in `sims1`; it passed.
- Clarified in Section 12 that current selection cuts are primary-sample based:
  they cut `r_input_p`, `Re_input_p`, and neighbour distance, but not
  `r_input_s` or `Re_input_s`. The secondary-size panel is labelled as
  uncut secondary `R_e`.
- Added an explicit global legend titled `Shear component` to the
  four-component profile figure.
- Re-ran the reduced Section 12 smoke in `sims1`; it passed.

## 2026-05-02

### Primary-Major-Axis Frame Selection MLP And Larger Slurm Requests

Replaced the always-nearest pair-frame default with a primary-major-axis frame.
The reference direction is now the primary galaxy intrinsic major-axis
orientation, so isolated rows no longer need an arbitrary nearest-neighbour
direction. Close-blend information is still included through `neighbored`-gated
secondary, distance, and pair-angle features.

Changed:

- Updated `SBS/sbs_shear/preprocessing.py`.
  - Added primary-frame feature engineering:
    `e_abs_p`, primary/secondary spin-2 ellipticity components, primary and
    secondary shear components, and pair-angle components in the primary frame.
  - Added `_blend` gated versions for distance, secondary properties,
    secondary shape/shear, and pair geometry.
- Updated `SBS/sbs_shear/selection_model.py`.
  - The default input feature list is now the 18-feature v6 primary-frame set.
- Updated `SBS/sbs_shear/coordinates.py`.
  - Selection gradients are exposed through the standard
    `dPsel_dgamma_parallel_*`, `dPsel_dgamma_cross_*`, and
    `dPsel_dgamma_perp_*` aliases using the primary-frame shear features.
  - Gated secondary-gradient aliases multiply by `neighbored`, converting
    feature gradients into physical shear gradients for non-blends.
- Updated SBS diagnostics and notebook defaults:
  - `SBS/scripts/evaluate_selection_blends.py`
  - `SBS/scripts/study_selection_gradients.py`
  - `SBS/scripts/validate_far_neighbor_invariance.py`
  - `SBS/scripts/study_selection_feature_importance.py`
  - `SBS/notebooks/inspect_selection_model.ipynb`
- Updated Slurm jobs in `SBS/jobs/`.
  - Selection training now requests 8 hours, 250G, 16 CPUs, 1 GPU, and 8 data
    workers.
  - Blend, gradient, and far-neighbour diagnostics now request 4 hours, 128G,
    12 CPUs, and 1 GPU.
  - Feature importance now requests 6 hours, 250G, 16 CPUs, 1 GPU, and 8 data
    workers.

Default v6 catalogue and checkpoint:

```text
catalogue: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather
checkpoint: SBS/models/selection_mlp_detected_v6_primary_frame.pt
```

Training result:

```text
job: 13852457
state: COMPLETED, exit code 0:0, elapsed 00:53:15, max RSS 16910220K
rows scanned: 105,206,950
rows after cuts: 94,536,670
sample rows: 4,000,000
temperature: 1.004671
validation:  logloss=0.270588, brier=0.072931, accuracy=0.917120,
             balanced_accuracy=0.917769, auc=0.937723
calibration: logloss=0.270948, brier=0.073107, accuracy=0.916698,
             balanced_accuracy=0.917370, auc=0.937795
```

Note: this training job had already started before the Slurm resource increase,
so it completed under its original active allocation. The pending diagnostics
were cancelled and resubmitted with the larger requests.

Diagnostics:

```text
job: 13852536
state: COMPLETED, exit code 0:0, elapsed 00:12:03
output: SBS/results/selection_blends_v6_primary_frame
all_after_cuts:    rows=1,000,000, observed=0.483877, mean_prob=0.484836,
                   logloss=0.270032, auc=0.938269
not_neighbored:    observed=0.520150, mean_prob=0.521636,
                   logloss=0.250068, auc=0.940517
neighbored:        observed=0.459724, mean_prob=0.460333,
                   logloss=0.283326, auc=0.935138
close_le_2arcsec:  observed=0.424839, mean_prob=0.426696,
                   logloss=0.299994, auc=0.929885
close_le_3arcsec:  observed=0.459724, mean_prob=0.460333,
                   logloss=0.283326, auc=0.935138
```

```text
job: 13852535
state: COMPLETED, exit code 0:0, elapsed 00:13:25
output: SBS/results/selection_far_neighbor_invariance_v6_primary_frame
not_neighbored mean |delta P|:
  all_blend_features:           0.000000
  distance_only:                0.000000
  secondary_properties:         0.000000
  blend_geometry:               0.000000
  secondary_shape:              0.000000
  secondary_shear:              0.000000
  primary_shear_relative_shape: 0.018729
close_neighbored_le_3 mean |delta P|:
  all_blend_features:           0.123544
  secondary_properties:         0.108345
  distance_only:                0.060091
  blend_geometry:               0.008962
  secondary_shape:              0.006999
  secondary_shear:              0.004176
  primary_shear_relative_shape: 0.019231
```

```text
job: 13852537
state: COMPLETED, exit code 0:0, elapsed 00:14:15
output: SBS/results/selection_gradient_study_v6_primary_frame
finite-difference checks:
  gamma_pframe_parallel_p:       corr=1.0, rmse=0.000211, mae=0.000126
  gamma_pframe_cross_p:          corr=1.0, rmse=0.000208, mae=0.000097
  gamma_pframe_parallel_s_blend: corr=1.0, rmse=0.000025, mae=0.000016
  gamma_pframe_cross_s_blend:    corr=1.0, rmse=0.000025, mae=0.000017
```

Validation:

```text
python -m py_compile SBS/sbs_shear/preprocessing.py SBS/sbs_shear/selection_model.py SBS/sbs_shear/coordinates.py SBS/scripts/evaluate_selection_blends.py SBS/scripts/study_selection_gradients.py SBS/scripts/validate_far_neighbor_invariance.py SBS/scripts/study_selection_feature_importance.py
PYTHONPATH=/home/z/Zekang.Zhang/SBS pytest -q SBS/tests/test_coordinates.py
3 passed
bash -n SBS/jobs/job_train_selection_mlp.sh SBS/jobs/job_validate_far_neighbor_invariance.sh SBS/jobs/job_evaluate_selection_blends.sh SBS/jobs/job_study_selection_gradients.sh SBS/jobs/job_study_selection_feature_importance.sh
notebook code cells parse OK
one-batch preprocessing smoke: all v6 features finite; all `_blend` features are zero for `neighbored=False`
```

Known caveat:

- The primary-major-axis frame is ill-defined for nearly round primaries. The
  fallback basis is finite and deterministic, but science interpretation of
  primary-frame shear gradients should be checked as a function of `e_abs_p`.

### Selection Notebook Review

Reviewed and updated `SBS/notebooks/inspect_selection_model.ipynb` for the v6
primary-major-axis-frame workflow.

Changed:

- Cleared stale saved outputs that still showed an older v3 checkpoint while
  the code pointed at the v6 model.
- Updated the opening notes and gradient interpretation:
  - `parallel` is aligned with the primary intrinsic ellipticity spin-2
    direction.
  - `cross`/`perp` is the 45-degree rotated spin-2 component.
  - Secondary physical gradients are `neighbored`-gated.
- Added a primary-frame input sanity section:
  - groups the 18 checkpoint features,
  - checks `_blend` features for `neighbored=False`,
  - reports the `e_abs_p` distribution and near-round fraction.
- Added a Slurm diagnostic summary section reading:
  - `SBS/results/selection_blends_v6_primary_frame/blend_subset_metrics.csv`
  - `SBS/results/selection_gradient_study_v6_primary_frame/finite_difference_autograd_checks.csv`
  - `SBS/results/selection_gradient_study_v6_primary_frame/gradient_by_distance.csv`
  - `SBS/results/selection_far_neighbor_invariance_v6_primary_frame/far_neighbor_shuffle_summary.csv`
- Updated binned performance plots to include primary-frame diagnostics:
  `e_abs_p`, primary-frame shear parallel component, neighbour distance, and
  pair angle relative to the primary major axis.
- Reduced local notebook autograd defaults and made the Slurm outputs the
  reference diagnostics.
- Added `Distance Gradients Split By Primary |e|`.
  - Reuses the gradients from the core distance-gradient cell.
  - Splits by fixed `e_abs_p` bins:
    `|e|<0.05`, `0.05<=|e|<0.15`, `0.15<=|e|<0.35`, and `|e|>=0.35`.
  - Plots distance-gradient curves for each primary ellipticity bin, with the
    no-neighbour population kept as a separate reference marker.

Validation:

```text
notebook code cells parse OK; outputs=0; nonnull_execution_counts=0
tiny CPU notebook smoke in conda env `sims1`:
  loaded v6 checkpoint,
  read a 2-batch capped-catalogue sample,
  ran primary-frame feature checks,
  computed local metrics,
  loaded Slurm diagnostic CSVs,
  built the binned-curve cell.
additional e_abs split smoke:
  loaded v6 checkpoint,
  read a 2-batch capped-catalogue sample,
  ran the core distance-gradient cell with reduced gradient rows,
  ran the e_abs split section successfully.
```

### Nearest-Neighbour Pair Frame And Gated Selection MLP

Built the next selection-model workflow around a nearest-neighbour pair frame
for every row while preserving `neighbored` as the indicator that the neighbour
was actually inside the close-blend/rendered radius.

Changed:

- Updated `blendemu/blendemu/response.py` and
  `blendemu/scripts/build_detection_catalogue.py`.
  - `retrieve_detection(..., attach_nearest_neighbor=True)` now fills
    secondary properties, pair angle, distance, and secondary shear for
    non-close-neighbour rows using the nearest non-self input object.
  - The default remains unchanged for blendemu callers unless the new flag is
    passed.
- Added
  `blendemu/jobs/job_sbs_detection_measurement_catalogue_nearest_skycos.sh`.
- Updated SBS selection features and diagnostics:
  - `SBS/sbs_shear/preprocessing.py`
  - `SBS/sbs_shear/selection_model.py`
  - `SBS/sbs_shear/coordinates.py`
  - `SBS/scripts/evaluate_selection_blends.py`
  - `SBS/scripts/study_selection_gradients.py`
  - `SBS/scripts/validate_far_neighbor_invariance.py`
  - `SBS/scripts/study_selection_feature_importance.py`
  - `SBS/notebooks/inspect_selection_model.ipynb`
  - relevant Slurm jobs in `SBS/jobs/`
- Updated `SBS/AGENTS.md` with the nearest-pair/gating reminder.

Nearest-neighbour detection catalogue:

```text
job: 13851961
state: COMPLETED, exit code 0:0, elapsed 00:06:03, max RSS 103365884K
output: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_nearest/sbs_detection_measurement_catalogue_train.feather
rows: 105,206,950
columns: 62
size: about 28 GiB
```

Validation of the new catalogue confirmed that `neighbored=False` rows have
finite nearest-neighbour columns. Sampled non-neighbour distances start just
above 3 arcsec, with medians near 3.97 arcsec.

First v4 pair-frame model:

```text
job: 13851989
state: COMPLETED, exit code 0:0, elapsed 00:44:19, max RSS 11216540K
checkpoint: SBS/models/selection_mlp_detected_v4_nearest_pair.pt
features: 17 direct nearest-pair features
temperature: 1.007966
validation:  logloss=0.267664, brier=0.071959, accuracy=0.918163,
             balanced_accuracy=0.918724, auc=0.939266
calibration: logloss=0.267910, brier=0.072063, accuracy=0.917978,
             balanced_accuracy=0.918563, auc=0.939259
```

The v4 far-neighbour validation exposed leakage from non-rendered neighbour
properties: for `not_neighbored_gt_6`, shuffling secondary properties changed
predictions by mean `|delta P| = 0.082862`. That is not acceptable for a
causal selection response, even though the classifier metrics were strong.

Second v5 gated-pair model:

```text
job: 13852178
state: COMPLETED, exit code 0:0, elapsed 00:48:36, max RSS 15270628K
checkpoint: SBS/models/selection_mlp_detected_v5_gated_pair.pt
features: 17
temperature: 1.011267
validation:  logloss=0.270349, brier=0.072820, accuracy=0.917408,
             balanced_accuracy=0.918017, auc=0.937699
calibration: logloss=0.270622, brier=0.072936, accuracy=0.917127,
             balanced_accuracy=0.917761, auc=0.937650
```

The v5 default feature list keeps primary pair-frame shear derivatives direct
and gates distance/secondary blend features by `neighbored`:

```text
Re_input_p_scaled, r_input_p_scaled, sersic_n_input_p,
e_parallel_p, e_cross_p, gamma_parallel_p, gamma_cross_p,
neighbored,
distance_scaled_blend,
Re_input_s_scaled_blend, r_input_s_scaled_blend, flux_ratio_blend,
sersic_n_input_s_blend,
e_parallel_s_blend, e_cross_s_blend,
gamma_parallel_s_blend, gamma_cross_s_blend
```

The v5 far-neighbour validation fixed the main leakage:

```text
job: 13852180
state: COMPLETED, exit code 0:0, elapsed 00:11:02
output: SBS/results/selection_far_neighbor_invariance_v5_gated_pair

not_neighbored_gt_6:
  distance_only mean |delta P|:            0.000000
  secondary_properties mean |delta P|:     0.000000
  secondary_pair_shape mean |delta P|:     0.000000
  secondary_pair_shear mean |delta P|:     0.000000
  primary_pair_orientation mean |delta P|: 0.019706
```

The remaining caveat is primary pair-frame orientation for isolated rows: the
nearest-neighbour direction is still an arbitrary frame when `neighbored=False`,
and shuffling primary pair-frame orientation changes predictions by about
`mean |delta P| ~ 0.02`. This is much smaller than the removed secondary
leakage but still worth revisiting, likely with rotation-invariant primary
shape/shear features or explicit orientation augmentation.

Close-blend and gradient diagnostics:

```text
job: 13852181
state: COMPLETED, exit code 0:0, elapsed 00:09:41
output: SBS/results/selection_blends_v5_gated_pair
all_after_cuts: logloss=0.269840, auc=0.938079
close_le_2arcsec: logloss=0.299501, auc=0.929719
close_le_3arcsec: logloss=0.282833, auc=0.935000

job: 13852182
state: COMPLETED, exit code 0:0, elapsed 00:10:55
output: SBS/results/selection_gradient_study_v5_gated_pair
finite-difference check: corr=1.0 for all checked pair-frame shear components
typical RMSE: 2.5e-5 to 3.2e-4
```

Validation:

```text
python -m py_compile SBS/sbs_shear/preprocessing.py SBS/sbs_shear/selection_model.py SBS/sbs_shear/coordinates.py SBS/scripts/train_selection_model.py SBS/scripts/evaluate_selection_blends.py SBS/scripts/study_selection_gradients.py SBS/scripts/validate_far_neighbor_invariance.py SBS/scripts/study_selection_feature_importance.py blendemu/blendemu/response.py blendemu/scripts/build_detection_catalogue.py
PYTHONPATH=/home/z/Zekang.Zhang/SBS pytest -q SBS/tests/test_coordinates.py
3 passed
bash -n SBS/jobs/job_train_selection_mlp.sh SBS/jobs/job_validate_far_neighbor_invariance.sh SBS/jobs/job_evaluate_selection_blends.sh SBS/jobs/job_study_selection_gradients.sh SBS/jobs/job_study_selection_feature_importance.sh blendemu/jobs/job_sbs_detection_measurement_catalogue_nearest_skycos.sh
notebook json/code cells parse ok
one-batch preprocessing smoke: all v5 features finite; all `_blend` features are zero for `neighbored=False`
```

## 2026-05-01

### Selection Feature Importance And Pruned V3 Retraining

Added a Slurm-backed feature-importance workflow before retraining the
coordinate-clean selection MLP.

Changed:

- Added `SBS/scripts/study_selection_feature_importance.py`.
  - Trains a pilot selection MLP using the current default feature list.
  - Computes individual and grouped permutation importance on held-out rows.
  - Saves CSV summaries, a plot, a README, and the pilot checkpoint.
- Added `SBS/jobs/job_study_selection_feature_importance.sh`.
- Updated `SBS/sbs_shear/selection_model.py`.
  - Default selection inputs were pruned from 29 to 26 features.
  - Removed low-importance candidates:
    `e1_input_p`, `relative_position_angle_sin2`, `e_cross_s`.
  - Kept all sky shear coordinates
    `gamma1_sky_p`, `gamma2_sky_p`, `gamma1_sky_s`, `gamma2_sky_s`
    even where single-feature classifier importance is small, because these
    are the derivative coordinates for `dP(s=1)/dgamma`.
- Updated `SBS/results/selection_feature_importance_v3/README.md` to record
  actual held-out rows used by the pilot importance pass.

Feature-importance Slurm job:

```text
job: 13841019
state: COMPLETED, exit code 0:0, elapsed 00:14:07, max RSS 5166088K
output directory: SBS/results/selection_feature_importance_v3
pilot checkpoint: SBS/models/selection_mlp_feature_importance_pilot_v3.pt
pilot rows: 1,500,000
importance rows: 225,000
baseline: logloss=0.276546, brier=0.074962, accuracy=0.913991,
          balanced_accuracy=0.914594, auc=0.936008
prune candidates: e1_input_p, relative_position_angle_sin2, e_cross_s
```

Top grouped importance by mean logloss increase:

```text
primary_size_flux:    0.876394
secondary_size_flux:  0.189426
pair_geometry:        0.117063
intrinsic_shapes:     0.016144
pair_aligned_shear:   0.009731
sky_shear:            0.008766
sersic:               0.003657
shape_pair_alignment: 0.003048
```

Production retraining Slurm job:

```text
job: 13841332
state: COMPLETED, exit code 0:0, elapsed 00:41:47, max RSS 14069212K
output checkpoint: SBS/models/selection_mlp_detected_v3_coord.pt
sample rows: 4,000,000
split: train=2,800,000, validation=600,000, calibration=600,000
early stopping: epoch 50
temperature: 1.003936
validation:  logloss=0.271022, brier=0.073036, accuracy=0.916960,
             balanced_accuracy=0.917601, auc=0.937477
calibration: logloss=0.271390, brier=0.073193, accuracy=0.916745,
             balanced_accuracy=0.917405, auc=0.937418
```

Validation:

```text
python -m py_compile SBS/scripts/study_selection_feature_importance.py SBS/scripts/train_selection_model.py SBS/sbs_shear/*.py
PYTHONPATH=/home/z/Zekang.Zhang/SBS python -m pytest -q SBS/tests/test_coordinates.py
3 passed
bash -n SBS/jobs/job_study_selection_feature_importance.sh SBS/jobs/job_train_selection_mlp.sh
checkpoint load smoke: 26 features, temperature 1.003936, val AUC 0.937477
```

### Blendemu Shear Convention Unified

Updated blendemu itself to use the same usual sky spin-2 convention as SBS:

```text
(q1, q2) = q(cos 2 theta, sin 2 theta)
```

Changed:

- `blendemu/blendemu/catalog.py`
  - `generate_catalog_realization(...)` now writes applied shear as
    `g1 = g cos(2 theta)`, `g2 = g sin(2 theta)`.
  - Generated catalogues carry
    `shear_component_convention = "sky_cos_sin"`.
- `blendemu/blendemu/response.py`
  - `e2ang(...)` now uses the usual half-angle
    `0.5 * atan2(e2, e1)`.
  - `spin2rot(...)` now projects spin-2 components into parallel/cross
    components with the same basis.
  - Detection, blending-response, and self-response catalogue builders write
    `shear_component_convention` into their outputs.
- `blendemu/README.md`
  - Added the shear coordinate convention note.
- `SBS/sbs_shear/coordinates.py`
  - `gamma1_input/gamma2_input` now map to sky shear by identity.
  - Missing convention metadata defaults to `sky_cos_sin`; there is no
    legacy component swap path.
- `SBS/SBI_shear.md`, `SBS/AGENTS.md`, and `SBS/tests/test_coordinates.py`
  were updated accordingly.
- `SBS/notebooks/inspect_detection_catalogue.ipynb` now includes
  `shear_component_convention` when present.
- Added `blendemu/tests/test_shear_coordinates.py`.

Validation:

```text
PYTHONPATH=/home/z/Zekang.Zhang/blendemu python -m pytest -q blendemu/tests/test_shear_coordinates.py
3 passed

PYTHONPATH=/home/z/Zekang.Zhang/SBS python -m pytest -q SBS/tests/test_coordinates.py
3 passed

python -m py_compile blendemu/blendemu/catalog.py blendemu/blendemu/response.py SBS/sbs_shear/*.py SBS/scripts/*.py
jq empty SBS/notebooks/inspect_detection_catalogue.ipynb SBS/notebooks/inspect_selection_model.ipynb
notebook code cells parse ok
```

Numerical interpretation:

- This is mathematically a coordinate relabeling if every generated component,
  angle conversion, response projection, model feature, and diagnostic is
  transformed consistently.
- It is not bitwise/numerically neutral for regenerated simulations with the
  same random seed: the per-object shear components and images are rotated
  relative to the old convention.
- Ensemble statistics with uniformly random shear angles should be unchanged
  within Monte Carlo noise.

Follow-up cleanup:

- Removed the SBS/blendemu legacy component-swap branch. Missing
  `shear_component_convention` metadata now defaults to the usual sky basis.
  This is the intended interpretation of the existing `gamma1/gamma2` columns:
  the old `sin/cos` generator changed the sampled angle variable, not the
  physical sky component basis consumed by the simulator.
- Updated `SBS/notebooks/inspect_selection_model.ipynb` to prefer the new
  regenerated catalogue path, with a temporary fallback to the old path while
  the Slurm rebuild is running.
- Updated `SBS/jobs/job_train_selection_mlp.sh` to train v3 from the new
  regenerated catalogue path.
- Added and submitted
  `blendemu/jobs/job_sbs_detection_measurement_catalogue_skycos.sh`.

Catalogue rebuild:

```text
job: 13840832
state: COMPLETED, exit code 0:0, elapsed 00:04:33, max RSS 103261016K
output: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos/sbs_detection_measurement_catalogue_train.feather
input simulation path: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/
rows: 105,206,950
columns: 62
size: 23.31 GiB
```

Additional validation:

```text
PYTHONPATH=/home/z/Zekang.Zhang/blendemu python -m pytest -q blendemu/tests/test_shear_coordinates.py
3 passed

PYTHONPATH=/home/z/Zekang.Zhang/SBS python -m pytest -q SBS/tests/test_coordinates.py
3 passed

bash -n SBS/jobs/job_train_selection_mlp.sh blendemu/jobs/job_sbs_detection_measurement_catalogue_skycos.sh
jq empty SBS/notebooks/inspect_selection_model.ipynb
notebook code cells parse ok

new catalogue light check:
record batches: 1,606
has shear_component_convention: True
first batch convention: sky_cos_sin
first batch shear_case: -0.1
```

### Coordinate Convention Cleanup

Made the orientation convention explicit and centralized for the SBS selection
workflow.

Changed:

- Added `SBS/sbs_shear/coordinates.py`.
  - Defines the SBS canonical sky spin-2 basis
    `(q1, q2) = q(cos 2 theta, sin 2 theta)`.
  - Maps `gamma1_input/gamma2_input` directly into canonical
    `gamma1_sky_*`, `gamma2_sky_*` features.
  - Provides pair-aligned spin-2 projections and shear-aligned gradient
    diagnostics in the same basis.
- Updated `SBS/sbs_shear/preprocessing.py`.
  - Intrinsic `e1/e2`, relative-position spin-2 features, canonical shear,
    and pair-aligned `e_parallel/e_cross` and `gamma_parallel/gamma_cross`
    are now generated from the same coordinate helper functions.
- Updated `SBS/sbs_shear/selection_model.py`.
  - Future default selection features now use canonical `gamma*_sky_*` and
    pair-aligned `gamma_parallel/gamma_cross` instead of raw blendemu shear
    columns.
- Updated `SBS/scripts/train_selection_model.py` and
  `SBS/jobs/job_train_selection_mlp.sh`.
  - Future default output is `SBS/models/selection_mlp_detected_v3_coord.pt`
    so the coordinate-clean model does not overwrite v2.
- Updated blend/gradient diagnostics and
  `SBS/notebooks/inspect_selection_model.ipynb`.
  - Existing v2 checkpoints are still supported; gradient diagnostics map raw
    shear gradients into canonical sky components before computing
    parallel/perpendicular components.
- Updated `SBS/SBI_shear.md` and `SBS/AGENTS.md` with the coordinate
  convention.
- Added `SBS/tests/test_coordinates.py`.

Validation:

```text
PYTHONPATH=/home/z/Zekang.Zhang/SBS python -m pytest -q SBS/tests/test_coordinates.py
3 passed

python -m py_compile SBS/sbs_shear/*.py SBS/scripts/*.py
jq empty SBS/notebooks/inspect_selection_model.ipynb
notebook code cells parse ok

catalogue smoke check: first record batch rescaled with all 29 default v3
features present
```

Known limitation:

- `selection_mlp_detected_v2.pt` was trained before this cleanup. It remains
  inspectable because diagnostics apply the chain-rule conversion, but the next
  production selection model should be retrained as the v3 coordinate-clean
  checkpoint.

### Selection Feature Update And V2 Training

Updated the selection MLP feature set after inspecting the first performance
notebook curves:

- Removed redshift from the current selection model inputs.
- Replaced axis-ratio/position-angle inputs with ellipticity components:
  `e1_input_p`, `e2_input_p`, `e1_input_s`, `e2_input_s`.
  - The measured catalogue already carries `e1_input_rot0_{p,s}` and
    `e2_input_rot0_{p,s}`; SBS maps those into the generic `e1/e2` names.
  - Older catalogues can still fall back to deriving ellipticity from
    axis ratio and position angle.
- Added relative-position geometry:
  `relative_position_angle_cos`, `relative_position_angle_sin`,
  `relative_position_angle_cos2`, `relative_position_angle_sin2`.
- Added shape-alignment features relative to the primary-secondary
  separation direction:
  `e_parallel_p`, `e_cross_p`, `e_parallel_s`, `e_cross_s`.
- Added `flux_ratio` and `neighbored` as explicit model inputs.
- Centralized raw-column dependency handling in
  `sbs_shear.preprocessing.raw_columns_for_selection_features(...)` so the
  training script and inspection notebook use the same feature engineering.
- Updated `SBS/SBI_shear.md` so the recommended selection model now treats BCE
  as the default probability-calibration loss, with focal loss as an optional
  fallback.

Training changes:

- `SBS/scripts/train_selection_model.py` now defaults to BCE loss because the
  science target is calibrated `P(s=1|x,n,\gamma)`, not just classification.
- Focal loss remains available through `--loss focal`.
- `load_selection_model(...)` now passes `weights_only=False` explicitly on
  current PyTorch versions, preserving checkpoint compatibility while avoiding
  the future-default warning during local inspection.
- `SBS/jobs/job_train_selection_mlp.sh` now launches the v2 GPU training job:
  `SBS/models/selection_mlp_detected_v2.pt`, 4M sampled rows, BCE loss,
  60 epochs, patience 10.
- The training script's default `--output` was also moved to
  `SBS/models/selection_mlp_detected_v2.pt` to avoid overwriting the earlier
  focal-loss checkpoint by accident.

Validation:

```text
python -m py_compile SBS/sbs_shear/*.py SBS/scripts/*.py
jq empty SBS/notebooks/inspect_selection_model.ipynb
notebook code cells parse ok
tiny CPU smoke train: completed 1 epoch and saved /tmp/sbs_selection_smoke.pt
v2 checkpoint load smoke: 25 features, temperature 1.01484
```

Completed Slurm retraining:

```text
job: 13834732
state: COMPLETED, exit code 0:0, elapsed 00:42:34, max RSS 11130124K
output checkpoint: SBS/models/selection_mlp_detected_v2.pt
raw rows scanned: 105,206,950
rows after cuts before sampling: 94,536,670
rows used: 4,000,000
selection rate: 0.4842
best validation BCE: 0.27101
temperature: 1.0148
Val: logloss=0.27097, brier=0.07309, accuracy=0.91679, balanced_accuracy=0.91743, auc=0.93765
Cal: logloss=0.27139, brier=0.07326, accuracy=0.91657, balanced_accuracy=0.91723, auc=0.93766
```

### Close-Blend Selection Diagnostics

Added diagnostics for model performance on close-neighbour blends:

- `SBS/scripts/evaluate_selection_blends.py`
  - Loads a reservoir sample from the full measured selection catalogue.
  - Reports metrics for all objects, non-neighbored objects, all neighbored
    objects, and close-neighbour cuts.
  - Computes autograd shear gradients only on close-neighbour subsets.
  - Writes CSV summaries and PNG plots.
- `SBS/jobs/job_evaluate_selection_blends.sh`
  - Slurm/GPU job for the blend diagnostics.
- `SBS/notebooks/inspect_selection_model.ipynb`
  - Added close-blend probability diagnostics for `distance <= 2 arcsec` and
    `distance <= 3 arcsec`.
  - Added close-blend-only autograd shear-gradient diagnostics.

Slurm diagnostic run:

```text
job: 13836188
state: COMPLETED, exit code 0:0, elapsed 00:06:26, max RSS 2791208K
model: SBS/models/selection_mlp_detected_v2.pt
sample rows: 1,000,000
gradient rows: 100,000 per close-neighbour cut
output directory: SBS/results/selection_blends_v2
```

Subset metrics:

```text
all_after_cuts:    rows=1,000,000, observed=0.483877, mean_prob=0.483526, logloss=0.270446, brier=0.072936, auc=0.938088
not_neighbored:    rows=399,710,   observed=0.520150, mean_prob=0.519266, logloss=0.251112, brier=0.065425, auc=0.940282
neighbored:        rows=600,290,   observed=0.459724, mean_prob=0.459729, logloss=0.283319, brier=0.077938, auc=0.935041
distance <= 2:     rows=335,035,   observed=0.424839, mean_prob=0.425898, logloss=0.299719, brier=0.084467, auc=0.929808
distance <= 3:     rows=600,290,   same as all neighbored because the current catalogue was built with r_max=3 arcsec
```

Close-blend autograd means:

```text
distance <= 2 arcsec:
  dPsel_dgamma1_input_p = -0.049654
  dPsel_dgamma2_input_p =  0.017982
  dPsel_dgamma1_input_s = -0.023327
  dPsel_dgamma2_input_s =  0.022731

distance <= 3 arcsec:
  dPsel_dgamma1_input_p = -0.041745
  dPsel_dgamma2_input_p =  0.022245
  dPsel_dgamma1_input_s = -0.021014
  dPsel_dgamma2_input_s =  0.018048
```

### Selection Gradient Study

Investigated whether the approximately zero-centered `dP/dgamma` histograms
are numerical noise or a real symmetry/cancellation effect.

Added:

- `SBS/scripts/study_selection_gradients.py`
  - Computes raw component gradients and shear-aligned
    parallel/perpendicular gradients.
  - Performs numerical finite-difference checks against autograd.
  - Reports observed/predicted selection rates by `shear_case`.
  - Writes gradient-distance summaries and plots.
- `SBS/jobs/job_study_selection_gradients.sh`
  - Slurm/GPU job for the gradient study.
- `SBS/results/selection_gradient_study_v2/README.md`
  - Documents output artifacts and interpretation.
- `SBS/notebooks/inspect_selection_model.ipynb`
  - Added a shear-aligned gradient section after the close-blend autograd
    cells.

Main conclusion:

- The raw `gamma1/gamma2` gradients can look centered near zero because the
  simulated shear directions are randomized.
- This is not an autograd numerical-noise issue: finite differences match
  autograd at correlation ~1.0.
- The shear-aligned `parallel` component is the more meaningful diagnostic for
  coherent response to shear amplitude.
- The `perp` component is close to zero and acts like a useful null direction.

Slurm gradient study:

```text
job: 13836536
state: COMPLETED, exit code 0:0, elapsed 00:06:10, max RSS 3183152K
sample rows: 1,000,000
gradient rows: 150,000 per subset, except distance <= 1 arcsec used 96,962 available rows
finite-difference rows: 20,000 on distance <= 2 arcsec
output directory: SBS/results/selection_gradient_study_v2
```

Finite-difference validation:

```text
component:gamma1_input_p  corr=0.9999997, rmse=2.64e-4
component:gamma2_input_p  corr=0.9999995, rmse=2.96e-4
component:gamma1_input_s  corr=1.0000000, rmse=2.44e-5
component:gamma2_input_s  corr=1.0000000, rmse=2.43e-5
parallel:p               corr=0.9999988, rmse=6.68e-4
parallel:s               corr=1.0000000, rmse=2.60e-5
```

Close-blend shear-aligned results:

```text
distance <= 1 arcsec:
  parallel p: mean=-0.066750, sem=0.002268
  perp p:     mean= 0.000633, sem=0.000992
  parallel s: mean=-0.010944, sem=0.001025
  perp s:     mean= 0.000574, sem=0.000806

distance <= 2 arcsec:
  parallel p: mean=-0.069130, sem=0.001767
  perp p:     mean=-0.001673, sem=0.000832
  parallel s: mean=-0.008531, sem=0.000876
  perp s:     mean= 0.000100, sem=0.000704

distance <= 3 arcsec:
  parallel p: mean=-0.070387, sem=0.001888
  perp p:     mean=-0.000202, sem=0.000753
  parallel s: mean=-0.001951, sem=0.000791
  perp s:     mean= 0.000950, sem=0.000625
```

Notebook follow-up:

- `SBS/notebooks/inspect_selection_model.ipynb`
  - Added a compact "Core Distance-Gradient Test" section.
  - It computes shear-aligned autograd gradients for real pairs binned from
    0 to 3 arcsec, then adds a separated no-neighbour reference point.
  - The no-neighbour point is not assigned a physical distance; the primary
    response is the clean comparison there, while neighbour/secondary response
    is only a placeholder diagnostic.

Validation:

```text
jq empty SBS/notebooks/inspect_selection_model.ipynb
notebook code cells parse ok
```

### Scientific Framing Update

- Updated `SBS/SBI_shear.md` to use a general selection event
  $s \in \{0,1\}$ instead of treating detection as the fundamental object.
- Added the central catalogue-density notation:
  $p_\text{cat}(\hat{x}|x,n,\gamma)
  \equiv p(\hat{x},s=1|x,n,\gamma)$.
- Clarified the factorization:
  $p_\text{cat} =
  p_\text{meas}(\hat{x}|x,n,\gamma,s=1)P(s=1|x,n,\gamma)$,
  where $p_\text{cat}$ integrates to the selection probability rather than
  one.
- Added the two downstream uses:
  response extraction with a target prior $p(x,n)$, and Bayesian shear
  inference with a prior over $p(x,n,\gamma)$ or $p(x,n)p(\gamma)$.

### Selection Bernoulli MLP

Added and trained the first concrete model for
$P(s=1|x,n,\gamma)$, with the current pilot target
$s=1 \equiv$ SExtractor detection:

- `SBS/sbs_shear/selection_model.py`
  - Selection-named MLP, focal loss, tabular preprocessor,
    temperature calibration, save/load helpers, and
    `probability_and_gradient(...)` returning `dPsel_d*` gradients.
- `SBS/sbs_shear/detection_classifier.py`
  - Kept as a backward-compatible alias layer for older detection-named code.
- `SBS/scripts/train_selection_model.py`
  - Streams Arrow/feather record batches from the large measured catalogue.
  - Applies SBS selection cuts and rescaling per batch.
  - Uses priority reservoir sampling so a bounded sample is drawn across the
    whole catalogue rather than from the first rows only.
- `SBS/jobs/job_train_selection_mlp.sh`
  - Slurm job for GPU training of the selection MLP.
- `SBS/AGENTS.md`
  - Updated the first model target to selection notation.

Pilot Slurm training:

```text
job: 13834280
state: COMPLETED, exit code 0:0, elapsed 00:14:36
catalogue: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_detection_measurement_catalogue_train.feather
raw rows scanned: 105,206,950
rows after cuts before sampling: 94,536,670
rows used: 2,000,000
selection rate: 0.4846
checkpoint: SBS/models/selection_mlp_detected.pt
```

Metrics after temperature calibration:

```text
Val: logloss=0.27210, brier=0.07357, accuracy=0.91584, balanced_accuracy=0.91651, auc=0.93746
Cal: logloss=0.27346, brier=0.07403, accuracy=0.91539, balanced_accuracy=0.91604, auc=0.93735
temperature: 0.3766
```

Post-training gradient smoke test:

```text
P(s=1) over 256 rows: mean=0.4462, min=0.0189, max=0.9566
gradient columns checked:
  dPsel_dgamma1_input_p
  dPsel_dgamma2_input_p
  dPsel_dgamma1_input_s
  dPsel_dgamma2_input_s
```

### Selection Model Inspection Notebook

Added:

- `SBS/notebooks/inspect_selection_model.ipynb`
  - Loads `SBS/models/selection_mlp_detected.pt`.
  - Shows checkpoint metadata and training curves.
  - Streams a bounded evaluation sample from the measured selection catalogue.
  - Reports log loss, Brier score, accuracy, balanced accuracy, AUC,
    confusion matrix, ROC/PR curves, and calibration curve.
  - Checks performance by shear case and neighbour status.
  - Plots observed versus predicted selection curves as functions of magnitude,
    size, distance, and neighbour redshift.
  - Summarizes residual structure and autograd shear-gradient distributions.

Validation:

```bash
jq empty SBS/notebooks/inspect_selection_model.ipynb
python - <<'PY'
import ast, json
from pathlib import Path
path = Path('SBS/notebooks/inspect_selection_model.ipynb')
nb = json.loads(path.read_text())
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        ast.parse(cell.get('source', ''), filename=f'{path}:cell-{i}')
print('code cells parse ok')
PY
```

Tiny runtime smoke test on one Arrow batch:

```text
rows: 2,048
logloss: 0.14654
auc: 0.98406
prob range: 0.01143..0.96049
```

### Catalogue Inspection Notebook

Added:

- `SBS/notebooks/inspect_detection_catalogue.ipynb`
  - Starts with catalogue loading and informative catalogue statistics before
    any classifier work.
  - Defaults to the SBS measured detection catalogue path and falls back to the
    original blendemu detection catalogue if the measured product is not built.
  - Includes schema metadata, bounded sample loading, basic integrity checks,
    shear/case coverage, detection and blending balance, SBS detection cuts,
    raw/scaled feature summaries, measured-property summaries when available,
    detection-rate curves, and redshift-aware blend diagnostics.
  - Uses Arrow IPC record batches for bounded sample reads from the large
    blendemu feather catalogue.
  - Leaves classifier loading, prediction, calibration, and gradient extraction
    for a later notebook step.

Removed:

- `SBS/notebooks/inspect_detection_classifier.ipynb`
  - Replaced by the catalogue-only notebook so the first inspection step stays
    focused on the training catalogue.

### Measured Detection Catalogue

Changed blendemu catalogue production in an opt-in way:

- `blendemu/blendemu/response.py`
  - Added `include_measured=False` and `measured_columns=None` to
    `retrieve_detection(...)`.
  - When enabled, joins SExtractor measured quantities and cross-match
    diagnostics onto detection rows:
    `match_id_detec`, `match_distance_pixel_cm`, `match_dmag_cm`, and
    `measured_*` columns such as `measured_mag_auto`, `measured_flux_auto`,
    `measured_flux_radius`, `measured_a_image`, and `measured_flags`.
  - Non-detected rows receive NaN measured values.
  - Default behavior remains unchanged, so the original blendemu
    `detection_catalogue_train.feather` path and schema are preserved.

- `blendemu/scripts/build_detection_catalogue.py`
  - Added `--include-measured` and `--measured-columns`.
  - If `--include-measured` is used without `--output`, the default output is
    `sbs_detection_measurement_catalogue_train.feather` rather than replacing
    `detection_catalogue_train.feather`.

- `blendemu/jobs/job_sbs_detection_measurement_catalogue.sh`
  - Slurm job for building the full measured detection catalogue.

Slurm build:

```text
job: 13833445
state: COMPLETED, exit code 0:0, elapsed 00:04:02
output: /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_detection_measurement_catalogue_train.feather
rows: 105,206,950
columns: 61
size: 22.91 GiB
```

The original default catalogue remains:

```text
/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/detection_catalogue_train.feather
columns: 33
size: 13.08 GiB
```

### Agent Resource Reminder

- Updated `SBS/AGENTS.md` to require Slurm jobs for resource-consuming work.
- Local commands are reserved for negligible edits, syntax checks, metadata
  inspection, notebook JSON validation, and tiny smoke tests.

Validation:

```bash
python -m py_compile blendemu/blendemu/response.py blendemu/scripts/build_detection_catalogue.py
jq empty SBS/notebooks/inspect_detection_catalogue.ipynb
python - <<'PY'
import ast, json
from pathlib import Path
path = Path('SBS/notebooks/inspect_detection_catalogue.ipynb')
nb = json.loads(path.read_text())
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        ast.parse(cell.get('source', ''), filename=f'{path}:cell-{i}')
print('code cells parse ok')
PY
python - <<'PY'
from pathlib import Path
import pyarrow.ipc as ipc
path = Path('/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_detection_measurement_catalogue_train.feather')
with ipc.open_file(str(path)) as reader:
    print('record_batches', reader.num_record_batches)
    print('columns', len(reader.schema.names), reader.schema.names[:8])
PY
```

Known limitation:

- The notebook reports bounded-sample diagnostics interactively; full-catalogue
  diagnostics should be run through Slurm rather than interactively.

## 2026-04-30

### Project Scope

- Kept SBS separate from `blendemu`.
- SBS code reads blendemu simulation/catalogue outputs as input data.
- `blendemu` should remain a data-producing dependency, not the home for this SBI/shear-calibration implementation.

### Planning Note

- Updated `SBS/SBI_shear.md` so the primary goal is a unified differentiable shear-calibration framework.
- Redshift-aware blending remains a key differentiator, but not the only scientific target.

### Detection Classifier Prototype

Added standalone SBS package files:

- `SBS/sbs_shear/detection_classifier.py`
  - PyTorch MLP detection classifier for `P(detected | true properties, neighbour properties, shear)`.
  - Uses smooth activations (`SiLU`, `GELU`, or `Tanh`).
  - Supports focal loss.
  - Stores a tabular preprocessor with mean/std scaling and missing-neighbour indicators.
  - Supports temperature calibration on held-out logits.
  - Provides `probability_and_gradient(...)` to compute calibrated `P(detected)` and raw-feature gradients such as `dP/dgamma1_input_s`.

- `SBS/sbs_shear/preprocessing.py`
  - Local SBS copy of the feature cuts and rescaling logic needed for detection-classifier inputs.
  - Avoids importing the `blendemu` Python package.

- `SBS/scripts/train_detection_classifier.py`
  - Trains the calibrated PyTorch classifier.
  - Saves model checkpoint plus boundary and train-curve diagnostics.

### Validation Run

Syntax check:

```bash
python -m py_compile SBS/sbs_shear/*.py SBS/scripts/*.py
```

One-case catalogue smoke test:

```bash
python blendemu/scripts/build_detection_catalogue.py \
  --data-path /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/ \
  --output /tmp/sbs_detection_onecase.feather \
  --cases 0 \
  --shears 0.0,0.1 \
  --n-jobs 1
```

Result:

```text
Saved 841,654 rows, 33 columns
```

Tiny classifier smoke test:

```bash
python SBS/scripts/train_detection_classifier.py \
  --catalogue /tmp/sbs_detection_onecase.feather \
  --output /tmp/sbs_detection_classifier.pt \
  --max-rows 2000 \
  --epochs 2 \
  --batch-size 256 \
  --hidden-dim 32 \
  --n-layers 2 \
  --device cpu
```

Result:

```text
Positive rate: 0.4800
Val: logloss=0.68426, brier=0.24557, accuracy=0.59667, balanced_accuracy=0.60978, auc=0.71557
Cal: logloss=0.68554, brier=0.24620, accuracy=0.58667, balanced_accuracy=0.60043, auc=0.72458
```

Gradient extraction smoke test succeeded for:

- `dP_dgamma1_input_p`
- `dP_dgamma2_input_p`
- `dP_dgamma1_input_s`
- `dP_dgamma2_input_s`

### Boundary Update

- Catalogue construction belongs in `blendemu`, not SBS.
- Removed the SBS catalogue builder after this boundary was clarified:
  - `SBS/sbs_shear/detection_catalogue.py`
  - `SBS/scripts/build_detection_catalogue.py`
- SBS now consumes a completed blendemu detection catalogue.
- Added `blendemu/scripts/build_detection_catalogue.py` for detection-only multi-shear catalogue construction.
- Updated `blendemu/blendemu/response.py` so detection catalogue rows include shear columns and `shear_case`.
- Updated `blendemu/blendemu/__init__.py` so `shape.py` imports on demand; this lets detection-catalogue building run without importing shape-measurement dependencies.

The multi-shear detection catalogue should be built from `blendemu`:

```bash
python blendemu/scripts/build_detection_catalogue.py \
  --config blendemu/configs/fs2_lsst_selec_emu.yaml \
  --output /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/detection_catalogue_train.feather
```

Full 50-case multi-shear catalogue built:

```bash
PYTHONPATH=/home/z/Zekang.Zhang/blendemu python blendemu/scripts/build_detection_catalogue.py \
  --config blendemu/configs/fs2_lsst_selec_emu.yaml \
  --output /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/detection_catalogue_train.feather \
  --batch-size 1 \
  --n-jobs 4
```

Result:

```text
Saved 105,206,950 rows, 33 columns -> /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/detection_catalogue_train.feather
```

Light verification:

```text
shears: each of -0.1, -0.05, 0.0, 0.05, 0.1 has 21,041,390 rows
cases: 0..49
detected: 49,196,121 true; 56,010,829 false
gamma1_input_p and gamma1_input_s span approximately [-0.1, 0.1]
```

Note: the blendemu builder also left per-case intermediate files named
`detection_catalogue_batch_*.feather` in the simulation output directory.
They are redundant after the merged catalogue is verified, but were left in
place rather than deleted automatically.

Train a pilot classifier:

```bash
python SBS/scripts/train_detection_classifier.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_selec_emu/detection_catalogue_train.feather \
  --output SBS/models/detection_classifier.pt \
  --max-rows 200000 \
  --epochs 20
```

### Documentation And Agent Reminder

Added:

- `SBS/WORKLOG.md`
  - Dated project log for implementation decisions, validation commands, and next steps.
- `SBS/AGENTS.md`
  - Scoped instruction file reminding future agents to keep SBS separate from `blendemu`.
  - Requires agents to update `SBS/WORKLOG.md` after substantive SBS changes.

### Next Tasks

- Train the detection classifier on the blendemu multi-shear detection catalogue.
- Add finite-difference validation against matched case/shear configurations.
- Decide whether the detection model should predict response to primary shear, neighbour shear, or both as separate reported quantities.

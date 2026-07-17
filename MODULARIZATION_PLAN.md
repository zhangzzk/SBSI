# SBSI Modularization Plan

Proposed refactor to make the response → inference flow (see `PIPELINE.md`) more modular and
navigable. **This pass touched nothing but safe `mv`s into `archive/`.** Everything below marked
*(proposed)* needs owner sign-off — several touch protected live-pipeline files and must not be
done autonomously. Nothing here has been applied except the "Done in this pass" section.

---

## 1. Done in this pass (files archived, no code edited)
9 scripts + 2 jobs moved to `archive/` / `jobs/archive/` (full manifest in `PIPELINE.md`):
finished toy-investigation cluster (7 `toy_*`), `measure_flow_c_train.py`,
`neighbor_shear_null.py`; jobs `job_toy_scan.sh`, `job_recov_blended.sh`. No live import or job
references any of them.

---

## 2. Duplicated logic → shared helpers *(proposed)*

The scripts share substantial copy-pasted logic. Highest-value extractions, in priority order:

1. **`response_ratio_diagnostic.py` is already the de-facto response library** (`model_mean_proj`,
   `_shape_target_indices`) imported by 8 scripts, yet it lives in `scripts/` under a "diagnostic"
   name. **Promote to `sbs_shear/response.py`** and re-export the old names for back-compat.
   *(PROTECTED file — sign-off required; keep a shim at the old path so nothing breaks.)*

2. **Catalogue IO + selection.** Feather streaming via `pyarrow.ipc` and
   `source_select_selection(DEFAULT_SELECTION_CUTS)` is re-implemented across
   `validate_constant_with_blend` (the canonical `load` / `CBASE`, already reused by
   `measure_flow_c`, `ood_rsim_check`, `calibrate_blend_residual_split`),
   `compute_response_target_blend`, `compute_response_target_constant`, `measure_gold_c`,
   `validate_allpairs_response`, `validate_constant_response`, `infer_posterior_shape`.
   → Extract a single `sbs_shear/io.py` (`load_catalogue`, chunked feather reader, standard
   selection) and have the validators/targets import it instead of each rolling their own.

3. **Lookup-attach.** Joining the S2 lookups (`g0_lookup`, `crowd_flux(_det)`, `meas_prim`,
   `ood_split`, `nn_dist`) onto a catalogue is duplicated in `validate_constant_with_blend`,
   `infer_posterior_shape`, `diagnose_additive_origin` (`attach_lookups`), and
   `build_mu_correction`. → `sbs_shear/lookups.py` with one `attach_lookups(df, which=[...])`.

4. **Response projection** `<e·ĝ>/g` (antithetic ±g secant, spin-2 projection) appears in
   `response_ratio_diagnostic`, `validate_allpairs_response`, `validate_constant_response`,
   `compute_response_target_blend`. → fold into `sbs_shear/response.py` from item 1.

5. **Path boilerplate.** Every script hardcodes
   `sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI"); sys.path.insert(0, ".../blendemu")`.
   → Add a `pyproject.toml`/`setup.cfg` so `sbs_shear` is `pip install -e .` and the two repos
   are on `PYTHONPATH` once (already the documented env contract) — then delete the per-file
   `sys.path` hacks. Removes ~3 lines × ~40 files and the absolute-path coupling.

6. **`infer_posterior_shape.py` (42 KB) doubles as a library** — `build_mu_correction` and
   `flow_response_by_mag` `import infer_posterior_shape as ips` for loader helpers/constants.
   → Split its reusable loaders into `sbs_shear/posterior_shape.py` (which already exists) or a
   new `sbs_shear/infer_io.py`; keep the script as a thin CLI.

---

## 3. Proposed directory layout *(proposed)*

Group `scripts/` (and mirror in `jobs/`) by pipeline stage so the ~46 scripts / ~150 jobs are
navigable. Protected files keep their basenames; only their directory would change (still
requires sign-off since they are protected against moves).

```
sbs_shear/            core library
  + response.py       (from response_ratio_diagnostic.py)
  + io.py             (catalogue load/stream/select)
  + lookups.py        (attach_lookups)
scripts/
  build/     build_detection_measurement_catalogue, augment_crowding,
             build_*_lookup, build_blend_lookup, build_blend_multiplicity,
             build_mu_correction, compute_*_target, compare_response_targets
  train/     train_measurement_model, train_selection_response
  harvest/   validate_constant_with_blend, validate_constant_response,
             validate_allpairs_response, plot_flow_calibration
  infer/     infer_posterior_shape, flow_response_by_mag, measure_flow_c, measure_gold_c
  probblend/ probblend_characterize, probblend_forward, probblend_calib(2d),
             probblend_ctx_diag, probblend_gap_diag
  additive/  diagnose_additive_origin, fit_additive_correction,
             finetune_additive_mean_head, apply_g0_mean_bias_shift,
             calibrate_blend_residual_split
  scene/     scene_coherent_model, scene_coherent_field, scene_ablation
  audit/     audit_blend_truth, audit_self_truth, map_truth_cases,
             match_fixed_sample, ood_rsim_check
jobs/        mirror the same subfolders
```

Caveat: archived scripts rely on `SBSI_ROOT = dirname(dirname(__file__))` (documented in
`archive/README.md`). If active scripts move one level deeper, that idiom and the hardcoded
`sys.path` inserts must be replaced by the packaging in §2.5 first — do §2.5 **before** §3.

---

## 4. Prioritized cleanup — safe actions

**Done in this pass** (see §1): toy cluster + 2 one-off diagnostics + 2 dead jobs archived.

**Proposed, needs owner sign-off:**

1. *(low risk)* Confirm-and-archive the **UNCERTAIN completed diagnostics** left in place:
   `measure_gold_c.py` (paired with the already-archived `measure_flow_c_train.py`) — archive
   once the additive-c/c2 work (WORKLOG cont.42+) is confirmed closed.
2. *(low risk)* Review the **ap7/np7-era job family** as a batch: `job_*_ap7.sh`,
   `job_np7_*.sh`, `job_snc_truemag_g0*.sh`, `job_build_allpairs_*.sh`,
   `job_blendlookup*_pilot/array.sh`. These predate the crowd/conc/meas_szfl pipeline. Left in
   place this pass (conservative — some scripts they call are still live). Owner to confirm which
   are truly superseded, then batch-move to `jobs/archive/`.
3. *(low risk)* `validate_constant_response.py` + its jobs (`job_const_validate_full.sh`,
   `job_constgold_lam300.sh`): superseded by `validate_constant_with_blend.py`. Archive once the
   pre-blend validator is confirmed unused.
4. *(low risk)* **Scene branch** (`scene_coherent_model/field`, `scene_ablation` + jobs): archive
   if the coherent-field ablation is closed; keep if still an open control per AGENTS.md.
5. *(medium risk)* Prune stale `models/*.pt` — ~80 checkpoints spanning the whole
   ap7→np7→crowd→conc→szfl→fixresp evolution (`models/` is ~600 MB). Only the current
   `*_fixresp_s50*` ensemble and the certified leaders are live. **Move, don't delete**, to a
   `models/archive/` after the owner tags the keepers. (Not touched this pass — outside the
   scripts/jobs remit and higher-risk.)
6. *(after §2.5)* Apply the shared-helper extractions and directory regrouping.

---

## 5. Dead code inside LIVE files (suggestions only — NOT edited)

No live pipeline file was edited. The large protected files
(`train_measurement_model.py` 48 KB, `validate_constant_with_blend.py` 38 KB,
`infer_posterior_shape.py` 42 KB) accreted many experiment-specific flags/branches across the
cont.1→47 history and are the most likely to hold now-unreachable code paths, but a concrete
dead-block audit was **not** performed in this read-only pass. Recommend, as a separate
sign-off task: run a coverage/vulture pass over these three plus `response_ratio_diagnostic.py`
and prune branches gated by retired CLI flags (e.g. superseded `--tag`/target-convention
options from the pre-`fixresp` era). Structural finding rather than dead code:
`response_ratio_diagnostic.py` and `infer_posterior_shape.py` are libraries masquerading as
scripts (§2.1, §2.6) — refactor, not deletion.

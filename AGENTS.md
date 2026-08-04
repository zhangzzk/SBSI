# Agent Instructions For SBSI

This directory is a standalone SBI/shear-calibration project.

## Read first: `CONVENTIONS.md`

`CONVENTIONS.md` fixes the vocabulary and the setup so any two tests are comparable: which catalogue
(constgold antithetic vs half-shear forward), the order population cuts are applied in, when to use
unsheared / sheared / measured shapes, true vs measured cuts, leg matching and what "both-detected"
silently removes, the response estimators, and the single definition of `m` and its relatives.
**Consult it before building a new number, and update it when a convention changes.**

Division of labour: `AGENTS.md` (this file) owns the MODEL, the seed convention, and the policy
rules. `CONVENTIONS.md` owns HOW quantities are constructed. On any overlap this file wins.

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
- **Seeds are part of the fiducial setup, split by flow OUTPUT: e (g1/g2) response = 16 seeds,
  flux/size = 4.** Ask what the reported number IS, not what the cut is on — a selection table cuts
  on flux/size but reports `m`, a bias on the SHAPE response, so it takes 16. Full statement and the
  worked example in "Ensemble Seed Convention" below.
- **Emulator:** blendemu tag `lsst_r_extnbr_indom_tuned` (73-trial Optuna search). Per-object lookup:
  `SBSI/results/blend_lookup_indomtuned_c40-139.feather` (`case`, `input_index`, `R_blend`).
- **constgold, in-domain, no cut, 16 seeds: `m = -0.123 +- 0.152%`** (std 0.608%),
  `R_sim = 0.8605`, `R_blend = 0.1358`, N = 11,674,408. **This is the number to quote** — it uses the
  whole in-domain population and carries seed error only.

  The near-domain table script used to report `-0.376 +- 0.152%` for the same quantity. That was
  never a second population: both paths land on the same galaxies (11,674,408 vs 11,674,409 rows — a
  one-row difference), and `source_select_selection` removes nothing in-domain (0.00% fail its
  distance condition; its mag/Re ranges are already subsumed by the emulator's own training cuts).
  The gap was the table's `--max-rows 4000000` SUBSAMPLE. Per-object response scatter is std 5.06
  against a mean of 0.86, so drawing 4M from 11.67M moves `R_sim` by an expected 0.00205 (observed
  0.00226) = 0.24%, the whole 0.25-pt gap.

  **CONFIRMED BY DIRECT TEST, and the default is fixed (2026-07-31, jobs 15385639 / 15389640).**
  `--max-rows` now defaults to `0` (whole population). On the full 11.67M with per-seed ratios the
  4-seed table gives **`-0.239 +- 0.430%`, matching the 4-seed dumps value `-0.239%` exactly** —
  where the subsampled version was off by 0.25 pt. Two independent paths, same number to three
  decimals. Do not reintroduce a subsample on the no-cut row: it is an absolute mean, so subsample
  noise hits it directly and exceeds the 16-seed error.

  **Correction (2026-07-31g): the subsample shift is NOT confined to the no-cut row.** An earlier
  version of this paragraph said cut rows "largely cancel it, which is why only the no-cut row was
  ever affected". That is wrong. `m` at a cut is `R_sim(cut)/R_model(cut) - 1`, and although both
  sides use the same draw they do not cancel: per-object response scatter is std 5.06 against a mean
  of 0.86, so the noisy draw moves `R_sim` while the smooth `R_model` barely moves. Removing the 4M
  subsample lifted EVERY row — `+0.03` to `+0.40` pt across cuts, `+0.25` at no cut. What genuinely
  cancels is `dm` and column (4) (model-vs-model), not the absolute `m` at a cut. This is the same
  distinction as the seed offset below, and for the same reason.

  The 4-seed value reads low relative to the 16-seed -0.123% because s503 is a -1.41% outlier —
  exactly why shape numbers use 16 seeds.

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
   0.7224`.

   **The "signature" wording here was WRONG and is retracted (2026-08-02, job 15483389).** This
   paragraph used to end "**A ~+5% in-domain m is the signature of using Gold-v1 out of its domain**,
   not an emulator fault." That is not a valid inference, and reading it that way cost real time: a
   run using the CORRECT V2 dom6x6 flow — 16 seeds, the fiducial dumps, verified by the glob in
   `plotting/plot_fid_flow_figures.py:206` (`ablate_s2c_lt500_dom6x6_perobj_s*.feather`) — produced
   **+4.666%**, indistinguishable from the +4.74% attributed above to Gold-v1 misuse.

   **A ~+5% m tells you the model total is ~5% below what the population needs. It does NOT tell you
   WHY.** Measured decomposition of that same −0.123% → +4.666% move (job 15485116,
   `scripts/diag_emudomain_subset.py`; identical dump rows throughout, `m` formed per seed):

   | population | N | R_sim | R_flow | R_blend | m |
   |---|---|---|---|---|---|
   | ALL (fiducial) | 11,674,408 | 0.8605 | 0.7258 | 0.1358 native | **−0.123%** |
   | KEPT by the emulator's pair cuts | 9,107,497 (78.0%) | 0.8605 | 0.6886 | 0.1649 native | **+0.830%** |
   | KEPT, emulator ALSO capped at 7" | 9,107,497 | 0.8605 | 0.6886 | 0.1336 | **+4.666%** |
   | DROPPED | 2,566,911 (22.0%) | 0.8604 | 0.8579 | 0.0325 native | **−3.358%** |

   Only **+0.95 pt of the 4.79-pt move is the population.** The other **+3.84 pt is the emulator being
   denied 19% of its own blend response** (0.1649 → 0.1336) by a 7"/neighbour-cut restriction it was
   never designed to run under — its native configuration is `r_max = 10"`, `k = 20`. **A restricted
   emulator is a HANDICAPPED emulator, not a differently-populated one**, so any model compared against
   a 7"-capped BlendEMU is being compared against a deliberately weakened baseline. Say so whenever
   quoting such a comparison.

   **RESTRICTING THE PAIR LIST HANDICAPS WHICHEVER MODEL RELIES ON THE EXCLUDED PAIRS — and that is
   usually NOT the model whose cuts define the list.** This is the sharpest trap in this section
   because it looks like the careful, confound-free choice. Worked example, all on the same dumps:

   | pair list | BlendEMU | flow #2 | gap |
   |---|---|---|---|
   | the EMULATOR's own cuts, 7" (9,107,497 rows) | +0.830% | +7.038% | **+6.208 pt** |
   | ALL pairs inside 7" (11,670,851 rows) | −0.126% native | **+0.572%** | **+0.698 pt** |

   Same models, same seeds, same catalogue; the verdict changes by a factor of nine. Scoring on the
   emulator's cut list removes exactly the faint/small neighbours flow #2 relies on, so it costs the
   flow 33% of the needed response and the emulator almost nothing. **Always score each model on the
   pair list it was TRAINED for, and report which list a number came from.** (Jobs 15485141 /
   15485170, WORKLOG 2026-08-02p and 2026-08-03a.)

   **Diagnose by checking `R_flow`, `R_blend` AND the row count against the fiducial (0.7258 / 0.1358
   on 11,674,408), not by pattern-matching the value of `m`.** Note especially that `R_sim` is a
   near-useless tell here: it sits at 0.8605 on the full set, 0.8605 on the kept 78%, and 0.8604 on the
   dropped 22%, so an unchanged `R_sim` is NOT evidence the population is unchanged.

   **The fiducial −0.123% is itself a CANCELLATION between two populations.** The blended majority
   under-predicts (+0.830% on 78.0%) and the weakly-blended remainder over-predicts (−3.358% on 22.0%);
   0.780 × 0.830 + 0.220 × (−3.358) = **−0.09%**, reproducing the fiducial value. On the dropped group
   the flow alone almost closes it unaided (`R_flow` 0.8579 vs `R_sim` 0.8604) and the emulator's
   0.0325 then overshoots. This is the same trap as the "Two traps" heading and as the in-domain
   cancellation recorded elsewhere: **do not read the small fiducial `m` as evidence that the model is
   right per-population, and do not evaluate a single-lever change against it alone.**

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

**The split is by WHICH FLOW OUTPUT drives the reported number, not by shape-vs-selection**
(clarified by the owner 2026-07-31; the earlier "shape 16 / selection 4" wording was too loose and
led to a 4-seed fig4 that should have been 16):

- **e (`measured_ngmix_g1/g2`) response -> 16 seeds.** Any multiplicative bias `m`, the certified
  pipeline number, any absolute shear response.
- **flux / size outputs -> 4 seeds.** Flux- and size-response validation
  (`scripts/eval_fluxsize_response.py`), proxy construction, cut-threshold calibration.

**Apply it by asking what the number IS, not what the cut is on.** A selection table cuts on measured
flux and size, so it is tempting to call it a 4-seed job — but the quantity it reports is `m`, a bias
on the SHAPE response of the surviving subset. The flux/size outputs only decide WHICH objects enter
the average; the average itself is an e-response. So the e-response standard binds: **selection
tables that report `m` need 16 seeds.** 4 seeds would only be enough for a table whose reported
number is itself a flux or size quantity.

Concretely, the constgold near-domain table at 4 seeds carried `+-0.43%` on its `m` column — nearly
3x the 16-seed error and too coarse to test against the `+-0.3%` target, which is the whole point of
the table. Its no-cut row read `-0.239 +- 0.430%` where the 16-seed value is `-0.123 +- 0.152%`: the
same quantity, the gap driven almost entirely by s503 (a `-1.41%` outlier carrying 1/4 of the weight
instead of 1/16).

- Default `SEEDS` in job scripts to the full 16 (`501 502 503 505 506 ... 517`; **504 does not
  exist**). The 4-seed set is `501 502 503 505`.
- Do not quote any `m` from 4 seeds.
- Scoring cost is linear in checkpoint count, so a 16-seed table is ~4x the GPU time. That cost is
  accepted for anything reporting `m`.

**The seed offset cancels in a DIFFERENCE, not in an absolute `m`.** `m` at a cut is model-vs-sim and
the sim side has no seed dependence, so each seed's own offset survives in full. It cancels only in
model-vs-model ratios (column (4)) and in `dm = m(cut) - m(no cut)`, where each seed's offset appears
in both terms. `dm` therefore comes out ~20x tighter (`+-0.02` to `+-0.16` vs `+-0.43`) and was
briefly plotted as a fig4 panel to work around the 4-seed noise; it is still computed and stored in
the npz but is no longer plotted, because at 16 seeds the absolute `m` is precise enough on its own.

### Which column the cancellation actually covers (2026-07-31)

Be precise about where the offset cancels, because it does NOT cancel in every "selection" column:

- **Cancels** — model-vs-model ratios `R_model(cut)/R_model(no cut)`, and the selection-induced
  excess `dm = m(cut) - m(no cut)`. Both terms move together seed to seed. 4 seeds is genuinely
  enough, and `dm` is the column to read.
- **Does NOT cancel** — the absolute `m` AT a cut. It is model-vs-SIM and the sim side has no seed
  dependence, so it inherits the full no-cut seed error (~0.43% at 4 seeds). Read it as a reference,
  not as a result; the absolute number comes from the 16-seed dumps.

**Errors must be formed per seed, not from ensemble means.** Building the ratio from ensemble means
and propagating the numerator's scatter alone (`sem(R_model(cut))/R_model(no cut)`) discards exactly
the cancellation above and returns the same `+-0.43%` on every row regardless of how aggressive the
cut is — a reliable tell that this bug is present. Form each ratio inside each seed, then take the
spread across seeds.

### Sign convention

`m = R_sim/R_model - 1` EVERYWHERE (AGENTS.md, fig3, the near-domain table). An earlier near-domain
table inverted it for the cut column only (`m_flow = R_model/R_sim - 1`) while the no-cut reference
stayed `sim/model - 1`, which mirrored the two about zero: the `R>0.30"` row keeps 99.99975% of the
population and so IS the no-cut case, yet printed `+0.247%` against a no-cut line at `-0.245%`,
making an exact null look like a 0.49-pt disagreement. Do not reintroduce the inversion.

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

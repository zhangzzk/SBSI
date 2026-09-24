## 2026-09-24 — Close-crowding flow inputs: choice, implementation, retrain launched

Owner request: fix the selection error below with half-shear data only.

**Choice of inputs (simulation only).** `scripts/crowding_feature_gain.py`
(CPU job 16677343; output `sbsi_caches/flow_joint_unbounded_20260922_v1/crowding_feature_gain.json`)
fits a gradient-boosted pass/fail classifier (MAG_AUTO < 25.8 and
FLUX_RADIUS > 3 px, g=0 leg) on 40 half-shear train cases (6.24M rows) and
scores 20 validation cases (3.12M rows).  Held-out log-loss (row-level
± 0.0003; paired differences are much tighter):
flow's 8 inputs 0.2869; + C at 1″ 0.2523; + C at 0.5/1/2″ 0.2463.  With the
three scales the crowded-minus-quiet residual (C1 quartiles) is within
± 0.005 in every true-r bin up to r = 28 (0.069 above r = 28), against
+0.13 to +0.37 with the 8 inputs alone.

**Inputs.** `nbr_close_050/100/200 = log1p(Σ_j F_j/F_i · exp(−d²/2s²))`,
s = 0.5, 1, 2″, over all rendered objects in the truth scene (same source as
`nbr_flux_*`); the same form as `sbsi.crowding.smooth_crowding` but in
physical arcsec over all objects.

**Implementation** (isolated checkout
`/project/ls-gruen/users/zekang.zhang/sbsi_flow_closecrowd_20260924`, a copy of
`sbsi_flow_restore_20260922`; the old checkout is untouched so earlier runs stay
reproducible):
- `sbsi/fixed_g0_domain.py`: `CLOSE_CROWDING_FEATURES` appended to
  `FLOW_FEATURES` (11 inputs); `PARENT_FEATURES` = first 8, the classifiers'
  contract (the frozen classifiers are not retrained).
- `scripts/build_close_crowding_lookup.py` (new): copies
  `crowd_flux_conc_c0-199.feather` and adds the three columns →
  `sbsi_caches/crowd_flux_conc_close_c0-199.feather` (job 16677431: 139,913,563
  rows, 200 cases, all identities matched; the 140 constant-shear scenes present
  have identical truth positions and fluxes, so the frozen check can reuse it).
- `prepare_full_domain_flow.py`: `--crowd-catalogue` joins the new columns
  (unmatched rows raise); `prepare_harmonized_flow_domain.py`: reads
  `FLOW_FEATURES[5:]` from the lookup.
- Classifier modules (`scene_classifier_joint`, `transported_classifier_joint`,
  `smooth_scene_classifier_features`) use `PARENT_FEATURES`;
  `evaluate_smooth_classifier_joint.py` passes the classifiers the first 8
  columns and the flow all 11.
- `scripts/extend_parent_close_crowding.py` (new): appends the columns to the
  evaluation parent → `constgold_truth_parent_closecrowd_20260924_v1/parent`.

**Retrain.** The 2026-09-22 nine-stage chain, cloned into
`sbsi_caches/flow_closecrowd_launchers_20260924/` with only the checkout, the
crowd lookup and the output roots changed (`full_domain_closecrowd_20260924_v1`,
`harmonized_closecrowd_20260924_v1`, `flow_joint_closecrowd_20260924_v1`).
Jobs: 0=16677431 1=16677439 2=16677440 3=16677441 4=16677442 5=16677443
6=16677444 7=16677445 8=16677446 9=16677447.  Frozen evaluation
(`sbsi_caches/joint_m_closecrowd_20260924_v1/`): parent 16677451, joint row
dumps 16677452/53, half-shear crowding check 16677454, decomposition 16677455,
one-case code smoke with the stage-2 flow 16677457.  Nothing is selected on
constant-shear data.

## 2026-09-24 — Half-shear only: the flow's selection cannot follow close-range crowding

Diagnostic only, half-shear data only (flow domain
`harmonized_secondary_unbounded_20260922_v1`, validation cases); no constant-shear
data used.  Truth crowding C_i = Σ_j F_j/(F_j+F_i)·exp(−d²/2·(1″)²) over every
rendered object within 4″ (truth catalogue; positions identical in both legs).

1. `sbsi_flow_restore_20260922/scripts/flow_response_vs_crowding.py` (GPU job
   16677237; output `sbsi_caches/flow_joint_unbounded_20260922_v1/flow_response_vs_crowding.json`):
   deployed flow vs simulation on 40 validation cases, 6.24M rows, by true r ×
   C quartile, 16 draws, case bootstrap (10000).  Flow-only m on these rows
   −0.48 ± 0.24 %.  Pass fraction sim/flow, least → most crowded quartile:

   | true r | q0 | q1 | q2 | q3 |
   |---|---|---|---|---|
   | 25–25.5 | 0.68/0.71 | 0.80/0.84 | 0.91/0.89 | 0.95/0.91 |
   | 25.5–26 | 0.39/0.42 | 0.54/0.65 | 0.77/0.73 | 0.90/0.78 |
   | 26–26.5 | 0.07/0.08 | 0.11/0.23 | 0.22/0.30 | 0.53/0.34 |
   | 27.5–28 | 0.11/0.08 | 0.07/0.08 | 0.12/0.10 | 0.53/0.16 |
   | > 28 | 0.15/0.12 | 0.13/0.14 | 0.42/0.20 | 0.85/0.26 |

   The flow's pass fraction is too flat in crowding.  Selection response at
   r 25–26 is too weak in the least crowded quartile (25.5–26 q0 sim
   −0.022 vs flow +0.002, ± 0.003).  Shape response agrees within a few percent
   except crowded bright galaxies (22–23 q3 sim 0.927 vs flow 1.001 ± 0.009;
   23–23.5 q3 0.769 vs 0.843 ± 0.009).
2. `scripts/crowding_information_check.py` (CPU job 16677286, simulation only,
   20 validation cases; output `.../crowding_information_check.json`): rows
   grouped by true r (0.5 mag) × 10 deciles of the flow input `nbr_flux_near`
   × 3 terciles of `nbr_flux_max`; inside each group, pass fraction of the
   most- minus least-crowded C quartile (row-weighted mean, case bootstrap 1000):
   r 23–23.5 +0.021 ± 0.002, 24–24.5 +0.053 ± 0.002, 25–25.5 +0.112 ± 0.001,
   25.5–26 +0.243 ± 0.002, 26–26.5 +0.304 ± 0.001, 27.5–28 +0.316 ± 0.013.
   Even at fixed flow crowding inputs, close-range crowding changes the pass
   probability by up to 30 percentage points.  The flow cannot learn this,
   because its inputs (log flux sums in 3″ and 3–7″, brightest neighbour)
   cannot tell a neighbour at 0.5″ from one at 2.9″, or a neighbour 10× brighter
   from one of equal flux at the same summed flux.

Reading: this is the "wrong crowded mix" term of the 2026-09-23 budget
(≈ 1.8 pp), now traced to a missing flow input and visible on the flow's own
half-shear data.  V3.6-like looked good because its true r < 26 population
drops the rows where the effect is largest (faint galaxies selected only
because a close neighbour lifts them over the cut).  Next: add a close-range,
flux-ratio-weighted crowding input (C at a few scales, from the scene catalogue
as the other neighbour inputs are) to the flow, retrain on half-shear data,
recheck with `flow_response_vs_crowding.py`; constant-shear data stays a final
frozen check only.  Limitation: C uses truth positions and fluxes; on real data
the input must come from the same scene prior the other neighbour inputs use.

## 2026-09-24 — Dropping test targets with bright close neighbours does not reduce the joint m

Owner question: does applying the emulator catalogue's bright-neighbour
rejection to the test population remove the bias?  `joint_response_decomposition.py`
gained `--reject-ratio/--reject-radius` (drop, on both sides, targets with a
truth object within the radius brighter than ratio × the target in true r
flux; the catalogue rule `blendemu.utils.remove_detection_w_bright_neighbour`
is 5× in 3″ on detections).  CPU jobs 16677088–93, outputs
`sbsi_caches/bright_reject_20260924_v1/decomp_{dep,gs}_{r5x3,r2x3,r1x2}.json`.

| rejection | targets dropped | m deployed flow | m per-magnitude-guard flow |
|---|---|---|---|
| none | 0 | +2.57 ± 0.27 % | +4.71 ± 0.27 % |
| >5× within 3″ | 21.4 % | +3.06 ± 0.28 % | +5.15 ± 0.28 % |
| >2× within 3″ | 36.3 % | +3.23 ± 0.28 % | +5.16 ± 0.28 % |
| >1× within 2″ | 27.4 % | +2.39 ± 0.27 % | +4.38 ± 0.28 % |

The strongly blended targets are where the model over-predicts (negative
contributions, see the entry below), so removing them does not lower m; the
r 24–26 per-bin contributions (+0.3 to +1.2 % each) are unchanged by every
variant.  The cut is applied in truth for this test only; on real data it
would use measured fluxes and its own shear response would need modelling.
Limitation: the variants share cases, so their differences are more precise
than the quoted ± (not separately bootstrapped).

## 2026-09-24 — Joint response gap by blending strength: two opposite errors, not one scale

Diagnostic only.  `sbsi_flow_restore_20260922/scripts/faint_blend_response_bins.py`
(CPU job 16676995) on the deployed-flow joint rows
(`joint_m_unbounded_retrain_20260923_v1/rows`, cases 40–119): galaxies passing
the simulation cut in both legs, binned by true r and by quintile of the stored
pair-summed blend response `r_blend`; per cell the shape response of the
simulation vs the model's flow and blend parts on the same galaxies.  Output
`sbsi_caches/neighbour_orientation_20260923_v1/faint_blend_bins.json`; plot
`SBSI/plots/joint_gap_by_blend_quintile.png`.  Case bootstrap (1000).

- The constant-shear simulations do not shear positions: both legs have
  identical RA/DEC and differ only in `g1`/`gamma1_input` (checked on case 48;
  MultiBand_ImSim `modules/ImSimObject.py` shears profiles in place).  The
  position-shift channel is therefore not part of this bias.
- Most-blended quintile: the model over-predicts at every magnitude, gap
  (sim − model) −0.21 ± 0.03 (r 21–22), −0.16 ± 0.02 (22–23), −0.10 ± 0.01
  (23.5–24), −0.04 to −0.07 ± 0.01–0.02 (r 24.5–27).  For bright primaries the
  simulation's response beyond the flow is about a third of the modelled blend
  term (r 22–23: 0.07 vs 0.23), so flow + pair-summed blend double-counts
  there.
- Middle quintiles at r 24.5–27: the model under-predicts, +0.03 to +0.14
  (± 0.01–0.04), including where `r_blend` ≈ 0 (r 25–25.5 q1: +0.074 ±
  0.008).  This is the flow's own faint-end shape response, not missing
  neighbours.
- Consequence: the joint m (+2.57 ± 0.27 % deployed, +4.71 ± 0.27 % with the
  per-magnitude guard) is a sum of cells with 5–20 % errors of both signs.  A
  global or per-magnitude correction of the flow cannot reach ~0.3 %; the
  composition of the flow response with the pair-summed blend response has to
  be fixed where blending is strong, and the flow's faint-end response where
  it is weak.
- Next (needs owner decision; touches the BlendEMU-trained response
  emulator): supervise the total response in crowded scenes directly, e.g. by
  training the blend term (or a correction to flow + blend) on pairs/scenes in
  which primary and neighbour are sheared together, and add the faint
  weak-blend cells to the flow's guard target.

## 2026-09-23 — Per-magnitude shape-response guard for the flow, interleaved with the density fit

The decomposition below showed the joint bias is mostly the flow's
cut-selected shape response being wrong in opposite directions at r < 24 and
r 24.5–26.  Fine-tune in the isolated checkout
`sbsi_flow_restore_20260922/scripts/`:

- `per_leg_guard_response.py`: `fit_per_leg_guard_response` and
  `PerLegGuardResponsePopulation` take optional `strata`/`n_strata`; the soft
  cut weights are split by a one-hot stratum, so the guard target and model
  response are per (cut, stratum) column.  `n_strata=1` reproduces the old
  behaviour.
- `refine_flux_size_response_flow.py`: `--guard-strata` (strata = truth r
  bins, edges 22, 23, 24, 24.5, 25, 25.5, 26, 26.5), `--guard-steps`,
  `--pair-weight` (0 skips the flux/size pair term), `--interleave-guard`
  (guard steps spread evenly through the density-fit epoch instead of in a
  burst at its end).  Validation prints `FLUXSIZE_GUARD` per-stratum
  primary-cut residuals, model − simulation of (e1g1+e2g2)/2.
- Burst guard steps (jobs 16675427/8, 16/64 steps) removed the per-stratum
  residuals (within ±0.007) but damaged the density fit: validation NLL
  −4.248 → −4.111 / −3.881, cut mass 0.2594 → 0.2571 / 0.2489 (sim 0.2588).
  Cancelled.
- Interleaved (jobs 16675894 i16, 16675895 i64; scale 0.1, lr as launcher):
  NLL stays −4.241 to −4.248, cut mass 0.2592–0.2600.  Start residuals
  [+0.023, +0.032, +0.030, +0.015, −0.004, −0.009, −0.010, +0.012, +0.019];
  i64 epoch 2: [+0.005, +0.004, +0.008, +0.003, −0.006, −0.005, −0.008,
  −0.003, −0.006].  Residuals move by ~0.01 between epochs, i.e. at the noise
  level of this validation check, so training was stopped after epoch 2.
  Selected `flow_joint_unbounded_20260922_v1/guard_strata_i64/epoch02.pt`.
- Joint m with that flow: rows jobs 16676037/8 in
  `sbsi_caches/joint_m_guardstrata_20260923_v1/` (same protocol as
  `joint_m_unbounded_retrain_20260923_v1`, flow path swapped by
  `make_rows_jobs.py`).  Decomposition (CPU job 16676042,
  `joint_m_guardstrata_20260923_v1/decomposition.json`):
  **m = +4.71 ± 0.27 %** (shape +4.36 ± 0.26, selection +0.35 ± 0.10),
  worse than the deployed flow's +2.57 ± 0.27 %.  Not adopted.
- Why: per-bin joint shape response, simulation / old flow / new flow:
  21–22 1.164/1.213/1.182, 22–23 1.125/1.173/1.139, 24–24.5
  0.799/0.784/0.769, 25–25.5 0.528/0.498/0.498, 25.5–26 0.454/0.431/0.426,
  26–26.5 0.474/0.445/0.415, 26.5–27 0.476/0.433/0.383 (sim ± 0.004–0.011).
  The guard removed the bright (r < 24) excess, which had been cancelling
  ≈ −1 % of m; it did not move the r 24–26 deficit (≈ 0.03 absolute), and the
  flow part at r > 26 fell (26–26.5: 0.074 → 0.045).  The per-leg soft-cut
  guard target measures shape + selection together, and at r 24.5–26 its
  start residual was only −0.004 to −0.010, so it had little to correct
  there.  Conclusion: the +2.6 % hid a larger faint-end deficit behind a
  bright-end excess; the faint end (r > 25.5, where the model response is
  mostly the blend term) is the real target.

## 2026-09-23 — Where the joint +2.6 % comes from: the flow's cut-conditional shape response by magnitude, then faint neighbour terms

Diagnostic only; nothing deployed changes.  Deployed-flow joint rows
(`joint_m_unbounded_retrain_20260923_v1/rows`, cases 40–119, flow-cut arm
with simulation usable flags).  New scripts in the isolated checkout
`sbsi_flow_restore_20260922/scripts/`; outputs and launchers in
`sbsi_caches/neighbour_orientation_20260923_v1/`.

- `joint_response_decomposition.py` (CPU job 16675261): exact per-bin split
  R = Σ w̄ Δe/2h (shape) + Σ Δw ē/2h (selection), simulation vs model.  The
  row dumps store the flow/blend shape for the cut as a sum over passing draws;
  it is divided by the cut mass here (an earlier run, 16675152, forgot this and
  is void).  Reproduces m = +2.57 ± 0.27 % (reference +2.49 ± 0.27; this
  version normalises the two legs jointly).  Shape +2.29 ± 0.25, selection
  +0.28 ± 0.10.  Per-bin total (shape+selection) contribution to m, %:
  21–22 −0.21, 22–23 −0.46, 23–24 −0.49, 24–24.5 +0.02, 24.5–25 +0.23,
  25–25.5 +0.53, 25.5–26 +1.05, 26–26.5 +1.06, 26.5–27 +0.44, >27 +0.41
  (per-bin ± 0.02–0.13).  Model shape response per selected galaxy splits into
  flow and blend parts: the blend part is 0.19 of 0.50 at 25–25.5 and 0.37 of
  0.44 at 26–26.5.  The r 24–26 shape deficit appears in every quartile of
  the neighbour spin-2 moment (least crowded quartile 24.5–25: +0.30 ± 0.06 %),
  so it is not a neighbour-geometry effect.
- The flow's own validation data (`flow_response_vs_mag.json`, secondaries
  sheared alone, cut applied) shows the same magnitude pattern for the cut-
  selected shape response, simulation − flow: 21–22 −0.040 ± 0.007,
  22–23 −0.032 ± 0.005, 23–23.5 −0.036 ± 0.005, 24.5–25 +0.014 ± 0.003,
  25–25.5 +0.021 ± 0.002, 25.5–26 +0.020 ± 0.003; it cancels to
  −0.48 ± 0.24 % over the own-data population.  Projected with the constant-
  shear bin weights it gives ≈ −1.0 % (r < 24) and ≈ +1.7 % (24.5–26), most
  of the joint's r < 26 terms.  Without the cut the flow's shape response is
  nearly right (0.4747 vs 0.4789 ± 0.0012; fine-tune metric 24–25.5 within
  +3.7/+0.5/−0.9 %), so the error is in how the flow's shape response co-varies
  with its predicted size/flux (the cut variables).
- `neighbour_orientation_selection.py` (CPU job 16675136): spin-2 neighbour
  moment N1 from truth positions (flux-ratio × Gaussian weights, 0.5/1/2″).
  Simulation measured e1 is pulled toward neighbours (slope at 1″: +1.4 to
  +1.8 for r 22–25.5, ± ≤ 0.02); the model's slope is ≈ 0 (it has no
  orientation input).  Pass flips under shear depend on N1 in the simulation
  (slope −0.3 to −0.8 bright, +0.02 to +0.07 at r > 26), ≈ 0 in the model.
  The selection response carried by N1 is −0.5 to −0.6 % of R (2″: −0.63 ±
  0.04 %), and it has the same sign pattern as the per-bin selection gaps
  (bright negative, r > 26 positive).
- V2.2 constgold-q3 self/deployed/other sims cannot check the secondaries:
  shapes were measured only for primary anchors (secondary anchors have
  NGMIX_G1 = −1), and the images were deleted.

Next: supervise the flow's cut-conditional shape response per magnitude on its
own paired g05 data, without empirical offsets, and then the r 26–27
neighbour/selection terms (neighbour orientation as a flow/classifier input).

## 2026-09-23 — Joint m with the flux/size-supervised flow: worse by +0.62 pp, from the shape response at r 24–25.5

Diagnostic of the fine-tune below; the deployed flow is unchanged.  Joint row
dumps with `flux_size_refine_s0.1_lr1e-6/epoch08.pt` and everything else as in
`joint_m_unbounded_retrain_20260923_v1/job_m_rows_{a,b}.sh` (GPU jobs
16672944, 16672945, 2.0 h / 2.2 h) →
`sbsi_caches/joint_m_fluxsize_ep08_20260923_v1/`.
`scripts/sim_flag_selection_test.py` now takes the rows directory and an
optional reference rows directory (default unchanged); with a reference it
reports paired changes on the same 10000 case-bootstrap weights.  CPU job
16673695 → `joint_m_fluxsize_ep08_20260923_v1/rows/sim_flag_selection_test.json`.

| arm (cases 40–119) | deployed flow | epoch08 | paired change (pp) |
|---|---|---|---|
| headline (classifier × flow cut) | +2.39 ± 0.27 | +3.01 ± 0.27 | +0.619 ± 0.005 |
| flow cut, simulation usable flags | +2.49 ± 0.27 | +3.11 ± 0.27 | +0.619 ± 0.005 |
| simulation pass flags, flow shape given passing | +0.65 ± 0.26 | +1.39 ± 0.26 | +0.735 ± 0.014 |
| simulation pass flags, flow shape over all draws | +4.26 ± 0.27 | +5.42 ± 0.27 | +1.163 ± 0.004 |

Per true-r bin, flow-cut arm, the change is the shape term at 24–25
(+0.286 ± 0.002) and 25–25.5 (+0.232 ± 0.002); selection-term changes are
≤ +0.036 per bin.  The fine-tune lowered the flow's shape response by ≈ 2 %
at r 24–25.5, where it was already too low (own data 24.5–25: 0.4686 →
0.4600 vs simulation 0.4828).  The guard term did not prevent it (its cuts
sit at magnitude 25.6–26 and it pools all galaxies).  The flux fix did not
move the joint's selection term.  Epoch08 is not adopted.

Next (running): the same fine-tune with the per-galaxy change of the
measured shape (g1, g2) added to the paired term (`--shape-scale`).  Measured
per-galaxy shape changes are noisy: the unscaled score constant is ≈ 5700 at
shape scale 0.02 (flux+size ≈ 37 at 0.1), so shape scales 0.1 and 0.25 are
being tried (jobs 16675000, 16675001; smoke 16674995 passed).

## 2026-09-23 — Flux/size shear-response term added to the flow: flux fixed, selection response not

The owner asked to supervise the flux (and size) shear response while keeping
the shape response (guard) and the density NLL.  Fine-tune of the deployed
flow `flow_joint_unbounded_20260922_v1/joint_probability_refinement_full/selected.pt`;
the deployed flow is unchanged.  All code is in the isolated checkout
`sbsi_flow_restore_20260922/scripts/`:

- `refine_flux_size_response_flow.py` — loss = NLL + guard (unchanged
  per-leg soft-cut shape-response term, 4 steps/epoch) + scale × paired
  flux/size term.  Paired term, per galaxy usable in both legs (simulation
  flags): with q = (ln FLUX_RADIUS, ln flux) and d = measured Δq between the
  sheared and unsheared legs, score = mean Σ_q (Δμ_A − d)(Δμ_C − d)/(4h²s²),
  where Δμ_A, Δμ_C are the flow's Δq from two independent latent replicas
  (8 CRN draws each), h = 0.05.  The two replicas make the score unbiased for
  the squared mean error; measurement noise adds only a constant.  Flow draws
  are floored at the smallest positive measured training value before the log
  (≈1.4k floored of 8×6.2M validation draws).  8192 pairs/step.  No empirical
  offsets; nothing is rescaled to the simulation.
- `nll_change_breakdown.py` — per-row NLL change vs the initial flow on 500k
  validation rows, by true r and measured flux/size decile, with standard errors.
- `plot_flux_size_finetune_comparison.py` — before/after plot.
- Launchers in `sbsi_caches/flow_joint_unbounded_20260922_v1/`:
  `job_flux_size_refine.sh`, `job_nll_breakdown.sh`, `job_eval_finetune.sh`.

Runs (A40; validation NLL on the 500k subset, initial −4.2484):

| job | setting | outcome |
|---|---|---|
| 16671538 | smoke | 3.7k pairs reused ~40× damaged NLL → pair batch raised to 8192 |
| 16671556 | scale 0.1, lr 1e-5 | NLL −4.197 after 1 epoch; cancelled |
| 16671557 / 16671656 | scale 0.03, lr 1e-5 | OOM on 16 GB GPU; rerun, NLL also degraded; cancelled |
| 16671913 | flux term off, lr 1e-5 (control) | NLL −4.197 as well → the damage is the 1e-5 learning rate with a fresh optimizer, not the flux term |
| 16671914 | **scale 0.1, lr 1e-6, 8 epochs** | **epoch08: NLL −4.2440, guard 0.049 (was 0.141), primary mass 0.2583 (target 0.2588)** |
| 16672712 | 8 more epochs from epoch08 | running; epochs 1–3 guard 0.07–0.09, flux unchanged |

Evaluation of epoch08 (job 16672708, own data, 16 CRN draws, per true-r bin,
validation; train agrees; the standard errors below are the model-side errors):

- Flux response d ln flux/d g+ — fixed.  26–26.5: before −0.048, after
  −0.075 ± 0.0001, simulation −0.075 ± 0.001; 26.5–27: −0.017 → −0.073 vs
  −0.081; 27–27.5: +0.052 → −0.034 vs −0.043; r < 25 excess halved (23.5–24
  +0.015 → +0.009 vs +0.008).  Whole sample −0.0096 → −0.0294 vs −0.0296.
- Size response — r < 21 0.456 → 0.259 ± 0.010 vs 0.225; r 21–26 unchanged
  within 1 %; 26.5–27.5 overshoots to −0.025/−0.028 vs −0.011/−0.005.
- Shape response (passing galaxies, cut size > 3 px, mag < 25.8) — mostly
  unchanged; whole sample 0.4747 → 0.4694 ± 0.0003 vs sim 0.4789.
- Selection response — **worse at the faint end**: 26–26.5 0.026 → 0.036
  ± 0.0004 vs 0.011 ± 0.003; 26.5–27 0.025 → 0.041 vs 0.013 ± 0.007.
  Pass fraction at 27–27.5 0.070 → 0.059 vs 0.099.  Own-data m:
  −0.48 ± 0.25 % → +0.32 ± 0.25 % (train −0.36 ± 0.13 → +0.44 ± 0.13 %).
- NLL change +0.0044 per galaxy: r < 22 improves (−0.030 ± 0.015,
  −0.009 ± 0.005); 22–23.5 +0.010–0.012; 26.5–27 +0.0065 ± 0.0005;
  27–27.5 +0.025 ± 0.002; > 27.5 +0.13 ± 0.01 (few galaxies).
- Plot: `/home/z/Zekang.Zhang/SBSI/plots/flow_fluxsize_finetune_before_after.png`.

So matching the average flux change is not enough: the flow now gets the
mean Δ ln flux right, but which galaxies cross the cut between the legs still
differs from the images (selection response 3× too high at r 26–27).  The
guard loss improves because it pools all magnitudes under soft cuts.

Next: joint m with the epoch08 flow, everything else as in
`joint_m_unbounded_retrain_20260923_v1` rows a/b (jobs 16672944, 16672945 →
`sbsi_caches/joint_m_fluxsize_ep08_20260923_v1/`).

## 2026-09-23 — Shear response of flux and size: flow vs simulation on the flow's own data

Diagnostic only; no model changed.  The flow's loss supervises only the shape
response (guard term under soft cuts); the shear response of measured flux and
FLUX_RADIUS is never a training target.  New scripts in the isolated checkout:
`scripts/flow_flux_radius_response_vs_mag.py` and
`scripts/plot_flow_flux_radius_response_vs_mag.py`.  Same population and CRN
draws as `flow_response_vs_mag.py` (rows usable in both legs by simulation
flags, forward |g| = 0.05 leg, 16 draws), no measurement cut.  Response =
d ln q / d g+, with g+ the shear along the true intrinsic major axis
(flux and size are spin-0, so they respond at first order only through g+);
also the isotropic mean Δ ln q.  GPU job 16671016 (A40, 10 min) →
`sbsi_caches/flow_joint_unbounded_20260922_v1/flow_flux_radius_response_vs_mag.json`.
Dropped: 0 simulation rows; ~61 (train) / 14 (validation) row-equivalents of
flow draws with non-positive flux or radius.
Plot: `/home/z/Zekang.Zhang/SBSI/plots/flow_flux_radius_response_vs_true_mag_own_data.png`.

Whole sample (train; validation agrees): flux sim −0.0307 ± 0.0002 vs flow
−0.0096 ± 0.0000; size sim +0.0875 ± 0.0002 vs flow +0.0988 ± 0.0003.

- Flux, faint end (the magnitude-cut region): flow far too weak / wrong sign.
  26–26.5: −0.048 vs −0.076 ± 0.001; 26.5–27: −0.017 vs −0.083 ± 0.001;
  27–27.5: +0.052 vs −0.048 ± 0.002; 27.5–28: +0.104 vs −0.032 ± 0.007.
  Bright end r < 25: flow +0.005–0.007 too high (e.g. 23.5–24 +0.016 vs
  +0.009 ± 0.001).
- Size: right to 3–5 % at r 21–26 (e.g. 24–24.5 0.176 vs 0.171 ± 0.001),
  too high at 26–27.5 (26.5–27 +0.007 vs −0.013 ± 0.001; 27–27.5 +0.028 vs
  −0.007 ± 0.002) and at r < 21 (0.45 ± 0.02 vs 0.226 ± 0.001, noisy draw tail).
- Isotropic mean shifts are ≲ 10⁻³ and agree except r < 21 size.

So the pass/fail of galaxies at the magnitude cut responds to shear wrongly
in the flow: where the images make faint galaxies lose flux when sheared
along their major axis, the flow's flux barely changes or grows.  Consistent
with the entries below (selection bias sits at r 26–27+).  Next: restore an
explicit flux/size shear-response term in the loss (by true r), and re-check.

## 2026-09-23 — Joint m with the simulation's own pass/fail flags

Diagnostic only; no model changed.  Direct test of the entry below: in the
joint "flags" arm only the usable flag comes from the simulation, while the
cut MAG_AUTO < 25.8 & FLUX_RADIUS > 3 px is applied to the flow's predicted
flux and size.  New script `sbsi_flow_restore_20260922/scripts/sim_flag_selection_test.py`
(CPU job 16670525, 1.5 min) re-weights the existing row dumps (cases 40–119)
→ `joint_m_unbounded_retrain_20260923_v1/rows/sim_flag_selection_test.json`.
R_meas = 0.6708.

- classifier-predicted usability × flow cut (headline; rerun job 16670835):
  R = 0.6551, m = +2.39 ± 0.27 % (reproduces the headline).
- simulation usability (detection, unique match, valid shape/flux/size) ×
  flow cut (flags arm): R = 0.6544, m = +2.49 ± 0.27 %.  Real detections do
  not help; the bias sits in the predicted cut.
- simulation pass flag per leg, model shape = flow mean over its own passing
  draws + blend shift: R = 0.6664, **m = +0.65 ± 0.26 %**.  (18,115
  sim-passed rows had no passing flow draw and used the all-draw mean.)
- simulation pass flag, flow mean over all draws + blend: m = +4.26 ± 0.27 %
  (passing draws have different shapes from the rest, so the shape must be
  taken given passing).

So the flow's cut choosing different galaxies than the images' cut
accounts for 1.84 of the 2.49 %, consistent with the R_blend-mix estimate
below (−1.79 ± 0.02 % on cases 40–59).  With the flow's cut, the model
selects 2.7–5.8 % too many galaxies at r 26–27 but 30–50 % too few at r > 27,
and at 25.5–27 the wrong ones (too few crowded).  Remaining +0.65 %, per bin (selection | shape):
bright r < 24 shape −0.86 (flow shape response too high there), 24–26
shape +2.12 (flow shape response too low, 3–12 % on its own data), 23–25.5
selection −1.27 and 26–27 selection +0.60 (shapes of galaxies whose pass/fail
flips between legs, which the model's mean shape does not reproduce).

Next: make the model's pass/fail depend on crowding the way the images do
(e.g. emulator shifts of flux and size, or a crowding-aware selection model);
then the flow's own shape response error at 24–26 is the leading term.

## 2026-09-23 — Why the joint is biased when flow and emulator are each fine: same cut, different galaxies

Diagnostic only; no model changed.  Question: flow (m −0.36 ± 0.13 % on its
own data) and emulator (label/prediction ≈ 1 on its own data) are each good,
yet the constant-shear joint gives m ≈ +2.5 %.  Are the same cuts and
selection applied?

**Cuts and selection are the same.**  All three datasets (flow training,
emulator training, constant-shear test) use the same detection, crossmatch,
usability rule, stamp, and cut MAG_AUTO < 25.8 & FLUX_RADIUS > 3 px applied to
each sheared leg separately.  The emulator's g0 cohort cut gives the same
answer as the per-leg cut (ratios identical), and dropping BlendEMU's
bright-neighbour rejection changes nothing (measured/predicted 0.993 ± 0.011).

**Cause: the cut picks different galaxies.**  In the images, a faint galaxy
with neighbours looks brighter and bigger, so crowded faint galaxies pass the
cut more often.  The flow sees only the target's truth plus three neighbour-flux
summaries, and the emulator changes shape only, so the model under-selects
crowded faint galaxies.  Mean lookup R_blend over selected galaxies,
simulation vs model (flags arm, cases 40–59, case bootstrap):
25–25.5: 0.207 vs 0.202 (1.027 ± 0.001); 25.5–26: 0.283 vs 0.266
(1.064 ± 0.001); 26–26.5: 0.436 vs 0.376 (1.158 ± 0.002); 26.5–27: 0.490 vs
0.398 (1.230 ± 0.004); ≤ 25: within 0.6 %.  The differences (0.017, 0.060,
0.092) match the sim-only unexplained neighbour-shape gap measured
independently (constant response − self response at ±0.02 − joint blend:
0.014, 0.063, 0.087).  Giving the model the simulation's mix moves m from
+1.80 ± 0.63 % to +0.02 ± 0.61 % on these 20 cases (shift −1.79 ± 0.02 %).
This is a counterfactual using simulation selection, not a fix.

Remaining budget (full 80 cases, m = +2.49 ± 0.27 %): wrong crowded mix
≈ −1.8 of it; neighbour-shear-induced selection the model lacks (a neighbour's
shear changes whether the target passes; training-sim value ×2 = 0.048 and
0.080 per galaxy at r 25.5–26.5 vs constant excess 0.048, 0.081) ≈ +1.5;
flow's own error −0.27 (shape +0.76, selection −1.03, cancelling);
rows usable in only one leg −0.48.

Ruled out: emulator nonlinearity (label/pred at shear 0.2, 0.05, 0.02, ±0.02
all 0.99–1.05 ± 0.01–0.05); target self-response step size (0.4904 / 0.4864 /
0.4880); a (1 − e²) factor in the blend shift (d e_true/dg = 1.0008);
the lookup's k = 20 neighbour cap (reproduced the stored lookup exactly;
keeping every neighbour within 10″ raises R_blend 0.1–0.3 % and m by
+0.07 ± 0.07 %); primary/secondary asymmetry (same neighbour counts 8.46
within 10″, same magnitude and size distributions).

New scripts (isolated checkout `sbsi_flow_restore_20260922/scripts/`):
`split_usability_selection.py` (job 16669649),
`emulator_linearity_selection.py` (16669773, 16670126; sims1 from blendemu),
`self_response_step.py` (16670088), `rblend_neighbour_cap.py`
(16670315–6; outputs `constgold_truth_parent_unbounded_20260922_v1/rblend_cap/`),
`rblend_cap_m.py`, `crowding_selection_mix.py`
(→ `joint_m_unbounded_retrain_20260923_v1/rows/crowding_selection_mix.txt`).
Plot: `/home/z/Zekang.Zhang/SBSI/plots/rblend_selected_mix_sim_vs_model_nocut_retrain.png`.

Limitations: the mix and cap numbers use cases 40–59 only (20-case m is
+1.80 ± 0.63 %, lower than the 80-case +2.49 ± 0.27 %).  Next: make the
model's selection see crowding properly — e.g. let the emulator (or a
neighbour-aware selection model) also shift flux/size, or condition the flow
on richer crowding inputs — and model the neighbour-shear-induced selection
term.

## 2026-09-23 — Flow vs simulation on the flow's own data, by true magnitude

Diagnostic only; no model changed.  New scripts in the isolated checkout:
`scripts/flow_response_vs_mag.py` (flow-only selection and shape response per
true-r bin, rows usable in both legs by simulation flags, forward |g| = 0.05
leg, hard cut MAG<25.8 & FLUX_RADIUS>3 px on each leg separately, 16 CRN draw
pairs) and `scripts/plot_flow_response_vs_mag.py`.  GPU job 16668856 (A40,
10 min) → `sbsi_caches/flow_joint_unbounded_20260922_v1/flow_response_vs_mag.json`.
Plot: `/home/z/Zekang.Zhang/SBSI/plots/flow_response_vs_true_mag_own_data.png`.

Whole sample (train 160 cases / 24.9M rows; validation 40 / 6.2M):
shape sim 0.4802 vs flow 0.4751 (validation 0.4789 vs 0.4747); selection sim
−0.0067 vs flow +0.0001 (validation −0.0064 vs +0.0001); flow-only m
−0.36 ± 0.13 % (validation −0.48 ± 0.24 %).

Per selected galaxy (train; validation agrees):

- Shape: flow too high at r 21–24 by 3–3.5 % (e.g. 22–23: 1.134 vs
  1.096 ± 0.003), right at 24–24.5, too low at 24.5–26 (−3 %, −7 %, −12 %:
  0.470 vs 0.484, 0.303 vs 0.325, 0.158 vs 0.179 ± 0.001).  Same pattern as the
  constant-shear likelihood (bright high, 24–26 low), so that pattern is the flow's.
- Selection: sim −0.008 to −0.014 at r 23–25.5; flow about 60–70 % of that at
  23–24.5, −0.0015 vs −0.0087 at 25–25.5, wrong sign at 25.5–26 (+0.006 vs
  −0.003 ± 0.001) and too positive at 26–27 (+0.026 vs +0.009 ± 0.002).
- Pass fraction: agrees to < 1 % up to r 27; beyond, flow passes about half as
  often (27.5–28: 0.106 vs 0.214; > 28: 0.167 vs 0.333).

Limitation: rows usable in only one leg are excluded, so this does not contain
the large positive faint selection response of the constant-shear test (entry
below), which may come from galaxies whose usability flips between legs.  Next:
split that constant-shear selection term into usable-in-both vs one-leg rows.

## 2026-09-23 — Shape response vs true magnitude, with the model split into flow and blend-emulator parts

Diagnostic only; no model changed.  In the isolated checkout,
`scripts/localize_nocut_bias.py` now also writes `shape_response_{measured,model_flow,model_blend}`
(bin share of total R and `_within` per selected galaxy); the plot script
(`scripts/plot_selection_response_vs_mag.py`) gained `--kind shape`.  CPU job
16668769 → `rows/localize_nocut_bias_v3.json` (m reproduces +2.3929 ± 0.2697).
Plot: `/home/z/Zekang.Zhang/SBSI/plots/shape_response_vs_true_mag_nocut_retrain.png`.

Shape response per selected galaxy, sim vs headline model (flow part + blend part):

- r < 23: sim 1.121–1.125 ± 0.006–0.008; model 1.153–1.173 (+3–4 %).
- r 23–24: sim 0.984 ± 0.006; model 0.999.
- r 24–25: sim 0.705 ± 0.004; model 0.685 (0.546 + 0.138), −3 %.
- r 25–25.5: sim 0.528 ± 0.004; model 0.500 (0.303 + 0.197), −5 %.
- r 25.5–26: sim 0.454 ± 0.004; model 0.434 (0.168 + 0.266), −4 %.
- r 26–27: sim 0.474–0.476 ± 0.006–0.010; model 0.443–0.451 (flow only
  0.03–0.07, blend 0.38–0.41), −5 to −7 %.
- r > 27: consistent within errors.

Shape part of m by magnitude: r < 24 −0.84 pp; r 24–26 +2.26 pp; r > 26 +0.35 pp.
So the shape deficit sits mostly at r 24–26, where flow and blend parts are
comparable; the data cannot say which is short.  Past r ≈ 25.5 the model
assigns most of the response to the blend emulator.

Scale check (CPU job 16668807, ad hoc): the model blend shape response per
galaxy equals the stored lookup `R_blend` (ratio 0.93–1.02 for r 22–27.5).  The
lookup (`constgold_truth_parent_unbounded_20260922_v1/rblend`, same emulator
sha256) sums ~16 neighbour pairs per galaxy, versus ~8.5 per primary in the
emulator's half-shear training table, so per-galaxy `R_blend` in SBSI is ~2× the
per-primary label sum plotted in the entry below; per-pair calibration is what
that entry tests.

## 2026-09-23 — Selection response vs true magnitude: the model has almost none for faint galaxies

Diagnostic only; no model changed.  In the isolated checkout,
`scripts/localize_nocut_bias.py` now also writes each side's selection response
per bin (`selection_response_{measured,model}` as the bin's share of total R,
and `..._within` per selected galaxy of the bin); new
`scripts/plot_selection_response_vs_mag.py` plots them.  CPU job 16668674
(1 min 43 s) → `sbsi_caches/joint_m_unbounded_retrain_20260923_v1/rows/localize_nocut_bias_v2.json`;
totals reproduce the entry below (m +2.3929 ± 0.2697).  Plot:
`/home/z/Zekang.Zhang/SBSI/plots/selection_response_vs_true_mag_nocut_retrain.png`.

Selection response per selected galaxy (simulation vs headline model; flags arm
in brackets), case-bootstrap ±:

- r 23–25: sim −0.018 to −0.019 ± 0.001; model −0.008 to −0.010 (−0.005 to −0.008).
- r 25–25.5: sim −0.013 ± 0.001; model −0.003 (+0.001).
- r 26–26.5: sim +0.062 ± 0.004; model −0.003 (+0.005).
- r 26.5–27: sim +0.106 ± 0.010; model +0.001 (+0.017).
- r 27–27.5: sim +0.18 ± 0.02; model +0.002 (+0.044).
- r > 27.5: sim +0.31 ± 0.05–0.07; model ≤ +0.004 (+0.05–0.06).

Whole-sample selection response: sim −0.0013, model −0.0054.  The two errors
partly cancel: bright/intermediate bins pull m down (−0.95 pp for r 23–25.5),
faint bins push it up (+1.36 pp for r > 26 headline, +1.17 pp flags arm).
Using the simulation's usable flags instead of the classifier raises the faint
model selection response only to about one fifth of the simulation's, so the
missing shear dependence is mainly in the flow's measured magnitude/size draws.

Next: check how the flow's MAG_AUTO/FLUX_RADIUS pass probability depends on
the shear-shape alignment for faint galaxies, against the simulation.

## 2026-09-23 — Blend emulator vs its own training data, by true primary magnitude: no magnitude trend within errors

Diagnostic only; no model changed.  New script in the isolated checkout
(`/project/ls-gruen/users/zekang.zhang/sbsi_flow_restore_20260922/scripts/emulator_rblend_vs_mag.py`)
rebuilds the `fixed_g0_m258_r060_v2` pair population with BlendEMU's own
training helpers (g=0 cohort MAG_AUTO<25.8, FLUX_RADIUS>3 px; valid shapes in
both legs), predicts every pair, sums pairs per primary (measured
`delta_et1/shear` vs predicted), and bins by true `r_input_p`.  Case-bootstrap
errors (10000, seed 20260923).  CPU job 16668489 (4 min 46 s), outputs
`sbsi_caches/fixed_g0_m258_r060_v2/rblend_vs_mag_v1/rblend_vs_mag.{png,json}`.

Counts: 149,750,534 pairs, 17,577,624 primaries; 21,378 pairs with invalid
shapes dropped (same as training); 581,078 cohort primaries (3.2 %) have no
neighbour pair (R_blend = 0 on both sides) and are not plotted.

- Overall measured/predicted: fitted cases 40–199 1.002 ± 0.009; held-out
  cases 0–39 1.004 ± 0.017.
- Per bin (fitted cases), r 24–27: ratios 0.99–1.01, each ± 0.016–0.05.
  r 27.5–29.5 (19.5k primaries): 0.86 ± 0.10 and 0.88 ± 0.15; held-out
  r 27–28: 0.86 ± 0.11, 0.74 ± 0.17 — a hint of over-prediction at the very
  faint end, 1–1.5σ per bin.  r < 22.5 is noise-dominated.
- Mean R_blend per primary rises from ~0.01 at r=22 to ~0.2 at r>26.
- The held-out total ratio (1.004 ± 0.017) differs from the training record's
  development slope (1.041 ± 0.0035) because the slope projects on the predicted
  vector and weights by predicted power; this plot is an unweighted per-primary
  mean.

Limitations: this is the emulator's pair population on the g=0 cohort, not the
SBSI measurement population after per-leg cuts, and does not test Möbius
composition.  Next: the same comparison in bins of true neighbour flux.

## 2026-09-23 — Localizing the no-cut m (+2.39 %): it is flow/emulator shape response, concentrated in faint and small galaxies, not the classifier

Development diagnostics on the run of the entry below (cases 40–119, cut
`mag25.8_radius_gt0.60`).  No model was changed.  Code lives only in the
isolated checkout `/project/ls-gruen/users/zekang.zhang/sbsi_flow_restore_20260922`
(not this repository):

- `scripts/evaluate_smooth_classifier_joint.py`: new optional `--dump-rows DIR`
  writes one `caseNNN.npz` per case with every truth parent's measured legs,
  usability, classifier probabilities and model mass/flow/blend moments.
  Without the flag the evaluator is unchanged.
- `scripts/localize_nocut_bias.py` (new): exact additive split of m into bins
  of truth properties, with selection/shape and flow/blend terms.
- `scripts/matched_pair_m.py` (new): m on matched pairs (usable and passing the
  cut in both shear legs).
- `scripts/flow_validation_g0cut_m.py` (new): flow-only m on the flow's own 40
  validation cases, cut on the g=0 leg only.

Uncertainties are 10000-replicate case bootstraps (seed 20260914).

### Reproduction

Row dumps: jobs 16662786/16662787 (A40), outputs under
`sbsi_caches/joint_m_unbounded_retrain_20260923_v1/rows/`.  Row sums reproduce
those runs' per-case statistics to 2.6e-14.  Against the earlier V100 run
16661106, m differs by −0.0003 pp (GPU nondeterminism near the cut thresholds,
about 2 draws per 91k per case).

### Where m sits (job 16662826, `rows/localize_nocut_bias.json`)

Contribution to the headline m (the bins sum exactly to +2.39):

| true r | share of selected | contribution (pp) | m within bin |
|---|---|---|---|
| < 24 | 22 % | −1.05 ± 0.14 | −2 % to −4 % |
| 24–25.5 | 48 % | +0.78 ± 0.21 | +1.4 %, +3.4 % |
| 25.5–26 | 19 % | +0.95 ± 0.13 | +6.7 % ± 1.0 % |
| > 26 | 11 % | +1.72 ± 0.10 | +20 % to +58 % |

- Size: galaxies with Re < 0.26″ contribute +3.2 pp (m within +6 % to +31 %);
  larger ones contribute −0.8 pp.
- Crowding (truth `nbr_flux_near`): the most isolated quarter contributes
  +1.32 ± 0.15 pp and the most crowded eighth −1.22 ± 0.08 pp.  The R_blend bins
  show the same sign pattern, but those bins use the emulator's own noisy
  prediction and are not used for conclusions.
- Classifier vs flags: `actual_usable_flags` (the simulation's own usability,
  no classifier) matches `modeled_smooth` bin by bin within about 0.05 pp.
  The classifier is not the source.
- Split: smooth arm selection +0.63 pp, shape +1.76 pp; flags arm selection
  −0.01 pp, shape +2.50 pp.
- At true r > 27 the flow passes the cut 30–54 % less often than the
  simulation does (per-bin count excess).  At 26–27 it passes 2.6–5.4 % more
  often.

### Matched pairs (job 16664689, `rows/matched_pair_m.json`)

7,137,706 rows are usable and pass the cut in both legs.

- Measured R: 0.67077 ± 0.00176 (per leg) vs 0.68395 ± 0.00169 (matched).
  Selection lowers the simulation's response by 0.0132 ± 0.0006.
- Model on the same rows, flow draws cut per leg: R = 0.67561.  The model's
  selection term is −0.0205 ± 0.0001.
- m on matched pairs is +1.28 % ± 0.25 %.  m_full = +2.39 splits into
  selection +1.07 ± 0.10 and shape +1.32 ± 0.25.
- Without cutting the flow draws, matched m is +4.69 % ± 0.26 %.  This is not
  a like-for-like comparison, because failing draws respond less.
- An exact version (draws passing in both legs) needs a GPU rerun and was
  not done.

### Flow alone on its validation cases (job 16665787, `flow_joint_unbounded_20260922_v1/validation_g0cut_m.json`)

Rows are usable in both legs; the cut is on the g=0 leg; forward |g| = 0.05;
16 CRN draws; no blend term.

| | R11 | R22 | mean |
|---|---|---|---|
| simulation | 0.4788 ± 0.0017 | 0.4793 ± 0.0014 | 0.4791 ± 0.0012 |
| flow (own g=0 cut) | 0.4820 ± 0.0004 | 0.4676 ± 0.0004 | 0.4748 ± 0.0003 |
| m | −0.65 ± 0.33 | +2.50 ± 0.31 | +0.90 ± 0.25 |

The flow's response is anisotropic (R22 about 3 % below R11).  The
simulation's is not.

### Stored validation metrics (no rerun)

- The flow's per-leg guard at radius > 3 and mag < 25.8 is +1.25 % (R11) and
  +0.99 % (R22) too high.
- The emulator's held-out amplitude slope (cases 0–39) is 1.041 ± 0.004: the
  emulator is about 4 % too low.
- The emulator was fitted on cases 40–199, and the flow's validation cases
  include some of 40–119.  The m cases are therefore not independent of
  either model.

### Next

- Faint and small galaxies: the flow under-produces cut-passing draws at
  r > 27, and the matched-pair shape response is too low.
- The flow's R11/R22 anisotropy.
- The exact both-leg matched rerun, if needed.
- Any joint flow+emulator fine-tune needs held-out simulations kept separate
  from cases 40–119.

## 2026-09-23 — Flow and 17-input classifier retrained with no truth cut: the no-cut m falls from +2.95 % to +2.39 % ± 0.27 %; the goal is still not met

The owner asked that neither the classifier nor the flow carry a truth cut,
matching the R_blend emulator, whose training sample is cut on measurements
only.  Both were retrained from the same code with only the input population
widened; everything else in the measurement is identical to the headline run
(16647912) of the entry below.

### Correction to the entry below

The entry below says the R_blend emulator `fixed_g0_m258_r060_v2` "is fit at
r<26".  That is wrong.  It has **no truth cut**: it is trained on measured
`MAG_AUTO < 25.8` and `FLUX_RADIUS > 3 px` only.  Its "extrapolated" population
string refers to that measured selection, not to a truth-magnitude bound.

### Result

Cut `mag25.8_radius_gt0.60`, 80 cases (40–119), seed 7301, 64 draws, same
R_blend tables, parent, pairs and geometry as 16647912.  The simulation side
is bit-identical between the two runs (all per-case sums equal), so the change
is a paired case-bootstrap difference (10000 replicates, seed 20260914):

| branch | H: r<26.5 flow, old classifier | new: no-cut flow, no-cut classifier | change (pp) |
|---|---|---|---|
| **modeled_smooth** (headline) | +2.950 ± 0.271 | **+2.393 ± 0.270** | **−0.557 ± 0.007** |
| actual_usable_flags (flow only) | +2.626 ± 0.271 | +2.494 ± 0.271 | −0.132 ± 0.009 |
| matched_usable | +3.226 ± 0.269 | +3.072 ± 0.269 | −0.154 ± 0.007 |
| modeled_usable | +1.210 ± 0.266 | +1.064 ± 0.266 | −0.146 ± 0.006 |
| modeled_control | +1.487 ± 0.267 | +1.470 ± 0.267 | −0.017 ± 0.006 |
| modeled_geometry | +1.854 ± 0.267 | +1.903 ± 0.268 | +0.049 ± 0.007 |
| modeled_transported | +2.171 ± 0.268 | +2.252 ± 0.268 | +0.081 ± 0.007 |
| modeled_coordinate | +2.179 ± 0.268 | +2.266 ± 0.268 | +0.086 ± 0.007 |

Measured response R11 = 0.67077 in both runs; the modeled response for
`modeled_smooth` rises 0.65154 → 0.65509.

- The no-cut flow alone (the `actual_usable_flags` branch, which uses the
  simulation's own usability flags and no classifier) moves m by
  −0.13 ± 0.01 pp.
- The rest of the headline change, about −0.4 pp, comes from the retrained
  17-input classifier.  Only `modeled_smooth` uses it; the other branches keep
  their frozen classifiers and move by less than 0.1 pp.
- **+2.39 % ± 0.27 % is ~9σ from zero and ~8× the ~0.3 % target.  The goal is
  not met.**  Retraining the flow and classifier on the emulator's domain
  explains about a fifth of the no-cut excess.
- ± values are the case bootstrap only; they exclude training-seed and
  latent-stream uncertainty (one seed each).

### What was run

- Re-measurement with no truth cut: `harmonized_secondary_unbounded_20260922_v1`
  (job 16659529, 200 cases × 349,784 truth rows).  **Bit-for-bit check:** all
  30,911,361 objects of the r<26.5 re-measurement are present, with identical
  shapes and flags, in all 200 cases.
- Flow: the same nine-stage chain as the r<26.5 retrain, with
  `--truth-magnitude-max 30` (the simulation floor is r = 28.99999, so this
  keeps every object; `inf` would break stage 1's strict JSON writer).  Jobs
  16659740–16659748.  Stage 2 trained on 24.9 M pairs.  Stage 9 selected
  step 3: validation NLL −4.2465 → −4.2456, guard loss 0.194 → 0.143,
  `eligible_for_joint_validation: true`, `production_accepted: false` (same
  flags as the r<26.5 flow).  Output
  `sbsi_caches/flow_joint_unbounded_20260922_v1/joint_probability_refinement_full/selected.pt`
  (sha256 da73a8db…).
- Classifier: prepare → transported geometry → smooth prepare → smooth train
  (jobs 16659896–16659899), with `--truth-magnitude-max inf` and cases 40–119.
  Seed, split and hyperparameters are unchanged; protocols are derived from the
  archived ones by `scripts/write_unbounded_classifier_protocol.py`, which
  records `derived_from`.  0 unmatched pair primaries.  Best tune BCE 0.18304 at
  epoch 98.  Output
  `sbsi_caches/scene_classifier_unbounded_20260922_v1/smooth_classifier_full/selected.pt`.
- The measurement: job 16661106, launcher
  `sbsi_caches/joint_m_unbounded_retrain_20260923_v1/job_m_nocut_retrained.sh`,
  whose diff against the headline launcher changes only `--flow`,
  `--classifier-smooth`, the output path and the GPU request.  Output
  `main_unbounded_retrained_flow_classifier.json` in the same directory.
- GPU stages were switched from A40-only to any single large GPU (they ran on a
  V100) by `scontrol update` on the queued jobs; the launcher files still say
  `a40`.  At most one SBSI GPU job ran at a time.
- All code ran from the isolated checkout `sbsi_flow_restore_20260922`; no
  repository code changed.  The three restored classifier scripts and the new
  protocol writer live only there.

### Next steps

- The remaining +2.4 % sits in the faint objects that the no-cut sample adds.
  Bin the measured and modeled responses by true magnitude (`r_input_p`) to see
  where the model departs, and whether the flow or the R_blend term carries it.
- Nothing is deployed; `configs/models_v3_6_like.json` is unchanged.

## 2026-09-22 (later still) — The measurement sample is cut on measurements only: a no-truth-cut parent chain, two hardcoded bounds exposed, and a third one found

The owner directed that no truth cut be applied to the sample — selection
belongs on the measurements, "as usual, 0.6" and 25.8"", and explicitly **no
ellipticity cut**.  Asked how far to widen the truth parent, the owner chose
**no truth cut at all**.  This entry records the chain built to that
specification.  **No `m` value is claimed here**: both measurements were still
running when this was written.  **Both have since finished; results are below
and the goal is not met.**

### Results

Cut `mag25.8_radius_gt0.60`, branch `modeled_smooth`, 80 cases, seed 7301:

| run | parent | flow | m (%) |
|---|---|---|---|
| archive Main80 | r<26 | r<26 | −0.199979 ± 0.263649 |
| control A (16647120) | r<26 | r<26 | **−0.199979 ± 0.263649** |
| control B (16647120) | r<26 | r<26.5 | **+1.129283 ± 0.266528** |
| headline (16647912) | **no cut** | r<26.5 | **+2.950293 ± 0.270648** |

Control A reproduced the archive **bit-identically across all 24 summary
entries**, so the restored chain carries no code drift and the other two
numbers are interpretable.

The headline is **+2.95 % ± 0.27 %**, 95 % CI [+2.41, +3.46].  The target was
~0.3 %.  The goal is **not met**, by roughly ten times the target and ~11σ from
zero.  Every branch moves the same way, so this is not a quirk of
`modeled_smooth`:

| branch | A r26/r26 | B r26/r26.5 | H nocut/r26.5 |
|---|---|---|---|
| modeled_usable | −1.140 | +0.157 | +1.210 |
| modeled_control | −1.122 | +0.179 | +1.487 |
| modeled_geometry | −0.936 | +0.368 | +1.854 |
| modeled_transported | −0.609 | +0.704 | +2.171 |
| modeled_coordinate | −0.604 | +0.709 | +2.179 |
| modeled_smooth | −0.200 | +1.129 | +2.950 |
| actual_usable_flags | −0.135 | +1.192 | +2.626 |
| matched_usable | +0.565 | +1.907 | +3.226 |

(all ± 0.26–0.27; `modeled_smooth` is the configured branch)

**Decomposition.**  −0.200 → +1.129 is the flow retrain alone
(**+1.329 ± 0.050 pp, 26.8σ**).  The two control runs share parent, pairs,
geometry, R_blend, classifiers and seed, and their `measured_response` came out
bit-identical (0.6850254024308094), so the measured-side error cancels in the
difference rather than adding; treating the runs as independent would give
±0.375 pp and understate the significance 7.5-fold.  +1.129 → +2.950 is the
population widening (**+1.821 ± 0.380 pp**, independent samples, so no
cancellation).

**Mechanism.**  The response decomposition:

| run | measured | model | flow | blend | blend share |
|---|---|---|---|---|---|
| A | 0.685025 | 0.686398 | 0.527344 | 0.159054 | 23.2 % |
| B | 0.685025 | 0.677376 | 0.518393 | 0.158983 | 23.5 % |
| H | 0.670767 | 0.651544 | 0.467695 | 0.183849 | 28.2 % |

Widening the parent adds ~11.9 % more selected objects per case
(81.5 k → 91.2 k, from `selection_fraction` × `parent_usable`, case 40), all
faint.  The measured response falls 2.1 % but the model's flow response falls
9.8 %.  Treating the shared objects' responses as unchanged — an
approximation — the added faint objects respond at ~0.55 in the data and
~0.43 in the model, i.e. **the flow underpredicts their response by ~21 %**.

The predictive diagnostics agree that the flow is off on this population:

| run | generated − observed selection fraction | total variation |
|---|---|---|
| A | +0.001110 ± 0.000118 (9.4σ over) | 0.0354 |
| B | +0.000415 ± 0.000119 (3.5σ over) | 0.0368 |
| H | **−0.003578 ± 0.000129 (27.8σ under)** | **0.0461** |

On the r<26 parent the flow slightly over-selects; on the no-cut parent it
under-selects at 27.8σ and its joint distribution match degrades.

**Caution on the obvious fix.**  The flow underpredicts the response (pushing
m up) while R_blend rises 15.6 % to a 28.2 % share (pushing m down).  The two
errors partly cancel.  Retraining the flow without also widening the emulator
could move m further from zero before it moves closer.

### The measured cuts were already correct

`scripts/evaluate_smooth_classifier_joint.py` applies the only cuts in the
measurement, per leg, to measured quantities:

    FLUX_RADIUS > 3.0 px  = 0.6 arcsec at 0.2"/px
    flux > 10**(-0.4*(25.8-30))  = MAG_AUTO < 25.8 at zero point 30

There is no ellipticity cut, only an `|e| < 1` validity test.  Nothing in the
evaluator was changed.  The truth cut lived one level up, in *parent* and
*geometry* preparation.

### Two hardcoded truth bounds exposed (isolated checkout only)

Both patches live in `/project/ls-gruen/users/zekang.zhang/sbsi_flow_restore_20260922`,
not in the repository, because `scripts/run_disk_inference.py::implementation()`
hashes every `sbsi/*.py` and editing the live worktree would invalidate
prepared inference shards.  Both defaults are unchanged at 26.0, so existing
invocations behave exactly as before.

- `scripts/prepare_constgold_truth_parent.py` — `magnitude_max=26.0` was an
  unreachable default with no CLI route.  Added `--truth-magnitude-max`,
  relaxed the finiteness guard to admit `+inf` while still rejecting NaN and
  non-positive bounds, and recorded `truth_analysis_cut` in the manifest.
- `scripts/prepare_scene_classifier.py` — a **third** hardcoded bound of the
  same class as the two corrected in `ad53755`:
  `or np.any(context[:, 3] >= 26.)` raising
  `'strict true-r<26 finite eight-input anchor required'`.  Same treatment.
  Pre-patch sha256 `06d33d9a92d5e8c8`, which matched the geometry manifest pin.

### Validation

- `tmp/test_truth_parent_bound.py` — bounds nest
  (26.0 ⊂ 26.5 ⊂ 27.0 ⊂ inf → 2031/2261/2518/4000 of 4000 synthetic rows),
  `inf` keeps every finite magnitude, a NaN bound is rejected.
- **Job 16647788 — byte-level regression on the geometry patch.**  Re-ran the
  *recorded* r<26 geometry for case 40 through the patched script at the
  default bound and compared against the 2026-09-18 record:

      recorded prepared_sha256 : b18bb7b57a850ba5bdfe60a1dfdde1ba59e8365b9fb77efdca626a27eba94d92
      fresh    prepared_sha256 : b18bb7b57a850ba5bdfe60a1dfdde1ba59e8365b9fb77efdca626a27eba94d92

  with `counts`, `anchor_sha256`, `pair_sha256`, `anchor_without_labels` and
  `unmatched_label_parents` all equal.  The patch is behaviour-preserving.
- **Cross-node determinism.**  Case 40 of the R_blend/pairs stage was built
  twice — smoke job 16647706 on `th-cl-rome01n1`, array task 16647732_0 on
  `th-cl-rome01n4` — and produced identical pair `sha256`.
- Every reused input was verified by hash against the recorded r<26 build
  before submission: classifier parent manifest `ee702664b89dcb26`, domain
  manifest `ce3d1add4c3744a7`, case-40 truth catalogue `6a03f441c3afa248`,
  emulator model `f04a19d375caa843`, emulator metadata `b3f251d59685e964`,
  guard flow `6755f4e00fce016c`.

### The chain as it ran

Output root `sbsi_caches/constgold_truth_parent_unbounded_20260922_v1/`.

| # | stage | job | elapsed | result |
|---|---|---|---|---|
| 1 | truth parent, no cut | 16647536 | 1:17 | 27,982,713 rows, 0 truth-cut drops |
| 2 | R_blend + pairs | 16647732 (8 tasks) | ~4:00 each | 398 M + 14 G, 450 M pairs |
| 3 | geometry | 16647852 | 2:48 | 80 cases, 450,046,601 pairs |
| 4 | headline measurement | 16647912 | running | r<26.5 flow, ~4.7 h projected |
| — | flow-only control | 16647120 | running | r<26 parent, both flows |

The headline was built from the frozen Main80 argument list with exactly 13
tokens changed (flow, parent, pairs, geometry, output, 8 R_blend shards),
asserted by count.  All five classifiers, the disk-response candidate and its
Möbius composition, `truth_parent`, 64 draws, seed 7301, h=0.02, axis 1 and the
80 cases are untouched: this is a population change plus the retrained flow,
**not** a new estimator.  The configured likelihood is unchanged.

### Three things worth recording about the chain

- The response pairs are **not** a BlendEMU product.
  `scripts/build_constgold_fixed_g0_blend_lookup.py --pair-output-root` emits
  them in the same loop as R_blend, from the same emulator call — which is why
  the evaluator can demand they reproduce the lookup to 1e-12.
  `blendemu/scripts/prepare_output_conditioned_response.py` builds the emulator
  *training* catalogue and is not in this chain.  No BlendEMU change was needed
  and none was made.
- `sbsi/models.py::load_emulator` never reads `flow_checkpoints`; it forwards
  only the emulator metadata, model and conditions to `BlendingPredictor.load`.
  The R_blend builder's `--measurement-model` is therefore provenance, not a
  model input, and was left at the recorded r<26 guard flow.
- Geometry must run **without** `--domain-root`.  Supplied, the script requires
  every anchor to carry a g0/g05 training label and raises otherwise, which the
  ~238 k unlabelled faint anchors would trigger.  Those arrays exist to *fit*
  classifiers; `scripts/scene_classifier_joint.py::classifier_frames` reads only
  `ids`, `base_context` and `geometry`, and the classifiers here are frozen.
  The manifest records `with_corrected_labels: false`.

### Dropped and unmatched rows

- Parent: **3** g0 keys absent from the parent (cases 105, 106, 118, one each)
  out of ~7.9 M, dropped and counted in `g0_keys_outside_truth_parent_dropped`.
- Geometry: **7** parents out of 27,982,713 have no neighbour pair at all.
  Their crowding features and R_blend are zero, which is the correct value for
  an isolated object, not a failure.  The recorded r<26 build had none.

### Limitations

- **Leading systematic.**  The R_blend emulator `fixed_g0_m258_r060_v2` is fit
  at r<26, and its own recorded population string already described it as
  "extrapolated beyond its fit cohort" there.  With no truth cut it extrapolates
  across the whole faint tail to the simulation's r=29.000 floor.  Mean R_blend
  over the parent rises 0.219 → 0.284 (+30 %), so the weight genuinely moves
  into the extrapolated region.  R_blend is 0.159 of a 0.686 total response
  (23 %), which makes this larger than the ±0.26 % statistical error unless the
  emulator is accurate to a few percent out there.  Training a wider emulator is
  BlendEMU's responsibility, not this chain's.
- The joint flow is retrained at r<26.5, not unbounded, so it extrapolates on
  the ~2.8 % of the measured-selected sample with true r>26.5.  This was the
  trade the owner accepted against a 0 % selection leak.
- Both patched scripts live only in the isolated checkout; the repository copies
  still carry the hardcoded bounds.
- The ~21 % flow underprediction on the added faint objects is an estimate from
  a two-component split of the aggregate responses, assuming the shared
  objects' responses are unchanged between runs.  It is a diagnosis, not a
  measurement; a per-magnitude-bin response comparison would measure it.
- Nothing is deployed.  `configs/models_v3_6_like.json` still names the r<26
  flow, and on the evidence here it should stay that way: the r<26.5 flow is
  worse at fixed population by +1.33 ± 0.05 pp.

### Next steps

- **Do not deploy the r<26.5 flow.**  It is worse than the deployed r<26 flow
  on the r<26 population, and does not rescue the no-cut population either.
- Measure, rather than infer, the flow's response error against true magnitude:
  bin the measured and modeled responses by `r_input_p` on the no-cut parent
  and find where the model departs.  That converts the ~21 % estimate above
  into a number and shows whether the failure is confined to r>26.5.
- Decide whether to widen the flow's own training population to match the
  parent.  Note the caution above: the flow and emulator errors currently have
  opposite signs, so widening the flow alone may not reduce |m|.  Widening the
  emulator is BlendEMU's responsibility.
- The retained r<26.5 flow chain, the no-cut parent chain and both measurement
  JSONs are preserved under
  `sbsi_caches/constgold_truth_parent_unbounded_20260922_v1/` and
  `sbsi_caches/joint_m_mag265_control_20260922_v1/`.

## 2026-09-22 (later) — The r<26.5 flow is retrained: nine stages complete, both stage-9 gates pass, and a launcher bug of mine cost one job

Continues the entry below.  The chain submitted there has finished.  The
retrained artifact is

    sbsi_caches/flow_joint_mag265_20260922_v1/joint_probability_refinement_full/selected.pt
    sha256 9babb174b9aad738c5dcd887a833628cb757e888389e5e516311e05e33a56767

It is **not** deployed: `configs/models_v3_6_like.json` is unchanged and still
names the r<26 artifact.  No bias number is claimed here — the two `m`
measurements are the next step, not this entry.

### The chain as it ran

| # | stage | job | elapsed | state |
|---|---|---|---|---|
| 1 | base training domain | 16642808 | 1:31 | COMPLETED |
| 2 | base guard flow | 16642889 | 39:47 | COMPLETED |
| 3 | harmonized domain | 16642973 | 12:07 | COMPLETED |
| 4 | control-variate reference | 16642979 | 12:00 | COMPLETED |
| 5 | control-variate refinement | 16642981 | 7:57 | COMPLETED |
| 6 | target banks + label audit | 16642978 | 4:59 | COMPLETED |
| 7 | per-leg gradient reference | 16642983 | 13:34 | COMPLETED |
| 8 | joint probability gradient | 16642985 → **16644387** | 5:02 | COMPLETED |
| 9 | joint refinement → artifact | 16642987 → **16644389** | 20:19 | COMPLETED |

Stages 4–9 ran strictly serially, never holding more than one GPU.

### The launcher bug

Job 16642985 (stage 8) failed after three seconds, exit 4, with
`ERROR: file or directory not found: tests/test_selection_probability_gradient.py`.
The cause was mine: I had written a pytest gate into the stage-8 and stage-9
launchers naming test files that do not exist in the restored checkout
(`test_selection_probability_gradient.py`, `test_per_leg_refinement_selection.py`,
`test_guard_selection_mass.py`).  Under `set -e` the job died before the audit
ran, and 16642987 then sat on `DependencyNeverSatisfied` and was cancelled.

Checking rather than simply deleting the line: **no restored test covers either
script.**  The only relevant test, `tests/test_guard_density_gate.py`, imports
`refine_guard_control_variate_flow` (stage 5) and nothing else.  So the gate was
replaced with an import check, and the functional gate left where it already
was — each launcher runs the audit on CPU in `--smoke` mode before the GPU full
run, and both scripts refuse a reference whose smoke flag, flow hash, domain
manifest or helper hashes differ from their own.

Before resubmitting, every command-line flag in both launchers was checked
against the scripts' `add_argument` lists; all matched.  Resubmitted as
16644387 with 16644389 chained `afterok`.  Both completed, stderr empty.

### The two stage-9 gates

Both are enforced only when `--smoke` is absent, so the full run exercised
them.  Values read from the audit outputs rather than inferred from exit zero:

| gate | threshold | r<26.5 (new) | r<26 (deployed) |
|---|---|---|---|
| `independent_half_direction_cosine` | ≥ 0.99 | **0.99965** | 0.99943 |
| `reference_bank_mean_gradient_cosine` | ≥ 0.99 | **0.9999962** | 0.9999987 |
| `strict_common_descent` (4 banks) | all true | true | true |

### Refinement result

| | r<26.5 (new) | r<26 (deployed) |
|---|---|---|
| baseline guard loss | 0.367922 | 0.456960 |
| selected guard loss | **0.109785** | 0.318071 |
| guard reduction | 70.2% | 30.4% |
| baseline full validation NLL | −4.398843 | −4.768710 |
| selected full validation NLL | −4.397954 | −4.768034 |
| best step | 5 of 6 | 2 of 6 |
| `production_accepted` | False | False |
| `calibration_claim` | False | False |

The NLL columns are **not comparable between runs**: the r<26.5 validation set
contains fainter parents the r<26 set never held, so a harder set scores worse
by construction.  Within each run the NLL is flat across refinement (the
density gate's job), and the guard loss is what the refinement moves.

`production_accepted: False` and `calibration_claim: False` hold for the
deployed artifact too; they are not a new failure introduced by the retrain.

### Ancestry, verified by hash

The lineage reconstruction checks out end to end and with identical structure
on both sides — each stage-9 protocol records its own stage-5 output as
`initial_flow_sha256`:

| run | stage 5 `selected.pt` | stage 9 `initial_flow_sha256` | match |
|---|---|---|---|
| r<26.5 | `af28f4ee38ee4710…` | `af28f4ee38ee4710…` | yes |
| r<26 | `fade8aebfb482534…` | `fade8aebfb482534…` | yes |

### Validation

- Both `selected.pt` load: 99 tensors, 1,390,218 parameters, all finite,
  byte-identical architecture (same `state_dict` shapes, same
  `model_config` / `condition_preprocessor` / `target_transform` keys).
- Stage 9 stderr empty; `FLOW9_JOINTREFINE265_COMPLETE` printed.
- Stage 8 `soft_mass_residual` 0.001318 (new) against 0.003451 (deployed).

### Limitations

- **No bias number is claimed.** The retrained flow has not been measured.
- The r<26.5 selected guard loss being much lower than the deployed one is not
  evidence of a better model: the two guard losses are computed on different
  populations, as the NLLs are. Only the matched-sample `m` measurement settles
  whether this model is non-inferior.
- Stages 8 and 9 have no unit-test coverage in the restored checkout. The smoke
  run and the scripts' own identity checks are the whole gate.
- The r<27 bounded-prior confound (10.2M vs 20.2M atoms) stays parked.
- The shape-augmentation arms are still not reproduced at 26.5, on the ancestry
  evidence above.

### Next steps

1. Measure `m` on the **old r<26 evaluation sample** with the new flow — the
   non-inferiority check that isolates the model change from the population
   change. This is the one that decides whether the retrain is safe.
2. Measure `m` on the **new r<26.5 sample** — the actual target.
3. Only then consider whether `configs/models_v3_6_like.json` should change;
   it is untouched so far.
4. The BlendEMU change to `scripts/prepare_output_conditioned_response.py`
   remains uncommitted for owner review.

## 2026-09-22 — The r<26.5 flow retrain: the deployed lineage is nine stages, two scripts had the old bound hardcoded, and the dell08 swap had to be repaired

Owner: "go ahead with the flow retrain, park the prior", then "chain the rest
and report when the flow is retrained".  This entry covers restoring the
retired flow chain, correcting two hardcoded truth bounds, establishing what
the deployed flow actually descends from, and submitting the chain.  The
retrained flow does not exist yet; the chain is running.

### What the deployed flow actually descends from

`configs/models_v3_6_like.json` deploys
`flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt`.
Its `protocol.json` records `initial_flow_sha256 = fade8aebfb482534…`, which is
the sha256 of `harmonized_secondary_20260917_v1/refinement/selected.pt` — the
control-variate refinement.  It is **not** either shape-augmentation arm
(`control` 1829b31cfeca33ab…, `augmented` a88af411e3d7875d…).  Those two arms
are siblings that share the deployed artifact's directory, not ancestors, so
they are not rebuilt.

Reproducing the deployed artifact at r<26.5 therefore takes nine stages, not
the six previously assumed; the three missing ones are the audits that produce
the reference each refinement optimises against.  Every producer below was
identified by matching a recorded sha256 against the archived sources, not by
directory or script name.

| # | stage | script | resource |
|---|---|---|---|
| 1 | base training domain | `prepare_full_domain_flow.py` | CPU |
| 2 | base guard flow | `train_full_domain_guard_flow.py` | 1 GPU |
| 3 | harmonized domain | `prepare_harmonized_flow_domain.py` | CPU |
| 4 | control-variate reference | `audit_guard_control_variate.py` | 1 GPU |
| 5 | control-variate refinement | `refine_guard_control_variate_flow.py` | 1 GPU |
| 6 | target banks + label audit | `per_leg_guard_response.py`, `audit_harmonized_usability_response.py` | CPU |
| 7 | per-leg gradient reference | `audit_per_leg_guard_gradient.py` | 1 GPU |
| 8 | joint probability gradient | `audit_joint_guard_probability_gradient.py` | 1 GPU |
| 9 | joint refinement → deployed | `refine_joint_guard_probability_flow.py` | 1 GPU |

Stages 4–9 are strictly serial, so the chain never holds more than one GPU,
inside the owner's two-GPU limit.  Queue state was checked as empty before the
first GPU submission.

### Files and behaviour changed

All source edits are in the isolated checkout
`/project/ls-gruen/users/zekang.zhang/sbsi_flow_restore_20260922`, built with
`git archive HEAD`, holding 19 restored modules and 4 restored tests, each
verified against `archive/research-2026-09-20/manifest.json` before copying.
Nothing was restored into the live worktree, because
`scripts/run_disk_inference.py::implementation()` hashes every `sbsi/*.py` and
restoring there would invalidate prepared inference shards.

- `scripts/prepare_full_domain_flow.py`: the population name is derived from
  `--truth-magnitude-max` rather than fixed at `lt26_v1`.  At 26.0 the name is
  unchanged.
- `scripts/prepare_harmonized_flow_domain.py`: the truth bound is **inherited
  from the baseline domain's manifest** instead of being hardcoded as `r < 26.`
  in four places (parent selection, population name, recorded
  `truth_analysis_cut`, limitations text).  The script already refused a
  baseline whose feature/target contract differed, so the bound belongs in the
  same check; this adds no new flag and makes the two domains incapable of
  disagreeing.  It raises if the baseline cut is not a finite strict
  `r_input_p` upper bound.  At a 26.0 baseline every output string is
  byte-identical to before.
- `scripts/train_full_domain_guard_flow.py`: the domain-validity check pinned
  `{"column": "r_input_p", "operator": "<", "value": 26.0}` and so rejected the
  26.5 domain as "not independently valid".  The magnitude literal becomes a
  column/operator/finiteness check; the three structural conditions it actually
  guards — no measured selection cut, independent per-leg validity, no anchor
  leg — are kept exactly as they were, and the accepted bound is now printed.
  The measured-magnitude guard thresholds `(25.6, 25.8, 26.0)` are a different
  quantity and are untouched: the primary selection stays `MAG_AUTO < 25.8`.
  Only the truth *parent* widens.
- `tests/test_harmonized_flow_domain.py`: the existing case is unchanged apart
  from passing the bound it always implied (26.), and a new case
  `test_widening_the_truth_bound_admits_the_objects_it_excluded` checks that a
  26.5 baseline admits the r=26.0 secondary a 26.0 bound drops while the role
  boundary and rendered-row-loss accounting stay put.
- `tmp/write_bank_receipt.py` (new, job-local): `per_leg_guard_target_banks.json`
  is written by no archived script — it is a receipt pinning the two bank files.
  Stage 9 checks exactly three of its fields, so the generator computes all of
  them from the artifacts on disk and refuses to pin banks that disagree with
  each other or with the domain they name.  Nothing is copied from the r<26 run.

### Repairing the dell08 swap

Stage 3 failed at case 1 with `FileNotFoundError` on a path inside
`harmonized_secondary_mag265_20260922_recheck`.  Cause: the script reads each
leg from `receipt['output']` — the absolute path the measuring job wrote — not
from `--measurement-root`.  Yesterday's swap moved the four rechecked cases'
shape catalogues into the main tree, so four receipts named files that were no
longer there.

The repair restores the file at the path its own receipt names rather than
rewriting any receipt: the receipt's claim is true, and it was the move that
made it unresolvable.  `tmp/repair_recheck_paths.py` copies only when the
candidate's sha256 already equals the recorded `output_sha256`, re-hashes the
restored copy, and refuses wholesale on any mismatch.  Restored 8 files
(cases 1, 64, 106, 182 × two shear legs), 8 verified, 0 problems.  Nothing was
deleted and the main tree was not modified.

The partial stage-3 output from the failed run (2 files, case 0 only, 15 MB)
was removed before resubmission.

### Validation

- `pytest tests/test_full_domain_flow.py tests/test_harmonized_flow_domain.py
  tests/test_guard_density_gate.py tests/test_flow_shape_augmentation.py -q`
  → **9 passed** in 5.87s, after the trainer patch.
- Stage 1 (job 16642808, CPU, 1:31, empty stderr):
  `FULL_DOMAIN_FLOW_PREPARED cases=200 g0=25,897,563 g05=25,890,802`, against
  19,601,305 / 19,597,500 at r<26 — **+32%**.  The manifest declares
  `truth_analysis_cut {r_input_p, <, 26.5}`, `independent_per_leg_validity
  True`, `measured_selection_cut None`, `anchor_leg None`.
- Stage 2 (job 16642889) trains cleanly; validation NLL improving through
  epoch 13, empty stderr.  Its first submission (16642869) failed instantly on
  the hardcoded-26.0 check described above.

### Limitations

- The retrained flow does not exist yet, so no bias number is claimed here.
- Stage 9's own gates (independent gradient halves cosine ≥ 0.99, reference
  bank agreement ≥ 0.99) have not yet been exercised on r<26.5 data; they are
  the point of the run and may legitimately fail.
- The r<27 bounded-prior confound (10.2M vs 20.2M atoms) is parked at owner
  request and remains unseparated.
- The shape-augmentation arms are deliberately not reproduced at 26.5, on the
  evidence above that they are not ancestors of the deployed artifact.  Any
  future claim that depends on them would need them rebuilt.

### Next steps

1. Let stages 3–9 complete; report when
   `flow_joint_mag265_20260922_v1/joint_probability_refinement_full/selected.pt`
   exists.
2. Then measure `m` twice: on the old r<26 evaluation sample, to isolate the
   model change as a non-inferiority check, and on the new r<26.5 sample, which
   is the actual target.
3. The BlendEMU change to `scripts/prepare_output_conditioned_response.py`
   remains uncommitted for owner review.

## 2026-09-22 — The r<26.5 emulator is retrained and the re-measurement is a clean superset; every discrepancy traces to the th-cl-dell0* nodes

Owner asked to retrain the flow and the R_blend emulator on a true r<26.5
parent instead of r<26 and re-verify the bias.  The emulator arm is complete.
The re-measurement the flow arm needs is complete and verified.  The flow
itself is not retrained yet.  A separate result, recorded below, is that the
r<27 bounded prior loses the positive-definiteness the bright stratum bought.

### Files and behaviour changed

- `blendemu/scripts/prepare_output_conditioned_response.py` (BlendEMU, left
  uncommitted for owner review): the hardcoded truth bound becomes
  `--truth-magnitude-max`, default 26.0, recorded in the manifest.  No other
  behaviour change.
- `harmonized_secondary_mag265_20260922_v1/code_snapshot/remeasure_common_optimizer_targets.py`:
  a copy of the frozen 20260917 snapshot (sha256
  `6096fd6953b60dbed6c742ebf5a1255be103e0cdabaad404da519a68611942b5`) differing
  only in that the truth bound is a parameter.  `diff -r` confirms the frozen
  `blendemu` package beside it is byte-identical.  The per-object seed formula
  is untouched, which is what makes the reproduction check below meaningful.
- Four measurement outputs replaced; see "The dell nodes" below.  Nothing was
  deleted: the replaced outputs are at
  `blendemu_runs/harmonized_secondary_mag265_20260922_dell08_quarantine/`.

### The emulator at r<26.5

`prepare_output_conditioned_response.py --truth-magnitude-max 26.5` then
`prepare_disk_response_moment.py` (job 16637497, CPU), then
`train_disk_response_moment.py` (job 16637595, CPU, 25:44).

| | r<26 | r<26.5 |
|---|---|---|
| fit rows | 128,809,729 | **167,785,890** (+30.3%) |
| kept rows | 161,012,399 | **209,730,224** |
| truth-cut dropped | 87,487,614 | **38,761,381** |
| base parameter | 0.008117317070720933 | **0.00902060410432565** |
| final train RMSE | — | 0.99702 over 218 rounds |

`baseline_unavailable_dropped: 0` and `invalid_label_dropped: 23,715`; no
unmatched rows were silently kept.  The faint labels already existed in
`response_catalogue_train.feather` (248,515,320 rows), so widening the bound
needed no new rendering or measurement on this arm.  Model at
`blendemu_runs/disk_response_moment_mag265_20260922_v1/model`.

### The re-measurement the flow needs

Unlike the emulator, the flow trains on a dedicated re-measurement whose
manifest limitation reads "only secondary true-r<26 ngmix columns
remeasured", so widening its domain does require new measurement — not new
rendering.  All 200 cases were re-measured rather than only the increment, so
every row comes from one script version and the reproduction check below is
possible.  Job 16637641 (`--array=0-199`, 8 CPUs each, throttle raised 24->40
mid-run), plus job 16641752 for the six-case recheck.

Truth parent per case: 110,695 -> 154,706 rows (+39.8%); matched 97,494 ->
129,202 (+32.5%).  Cost ~560 CPU-hours, against ~378 for the original r<26
campaign.  An increment-only run would have cost ~135, and was rejected in
favour of the single-version guarantee.

### It is a clean superset

Because the seed is keyed per object and is identical at all amplitudes, every
identity measured under r<26 must return bit-for-bit identical shapes and
flags under r<26.5.  Checked, not assumed (`verify_remeasure.py`, job-local):

**198 of 200 cases reproduce every stored value exactly.**  The two exceptions
are cases 77 and 85, and they are explained below rather than tolerated.

### The dell nodes

The first pass reproduced only 194 of 200.  The six failures were not random:
97.67% of their disagreeing entries differ by less than 1e-6 relative, which is
float32 rounding, and only 3,433 entries (0.0044% of the whole 200-case
parent) differ by more than 1e-3.  The disagreements are spread evenly through
each file, not clustered by worker.

Cross-tabulating case against the node that measured it settles it:

| campaign | node | cases run | cases disagreeing |
|---|---|---|---|
| new (16637641) | th-cl-dell08 | 4 | **4** |
| new (16637641) | every other node | 196 | 2 |
| old (16556786) | th-cl-dell07 | 1 | **1** |
| old (16556786) | th-cl-dell08 | 1 | **1** |
| old (16556786) | every other node | 192 | 0 |

The two stragglers in the new run are cases 77 and 85 — exactly the two cases
the *old* campaign happened to run on dell07 and dell08.  Every discrepancy in
either campaign traces to those machines and there is no unexplained residual.

Repeating the four new dell08 cases with `--exclude=th-cl-dell08` (job
16641752) reproduced the stored 20260917 values exactly: 0 of ~443,000 shape
entries differ, against ~27,500 for the dell08 run.  Those four outputs were
therefore swapped in and the dell08 versions quarantined.  For cases 77 and 85
the new pipeline produced identical results on two independent nodes each
(rome01n4 then rome07n1; rome04n3 then rome09n1) while the stored value stands
alone, so there the *old* value is the defective one and the new is kept.

`--constraint=x86-64-v3` does not prevent this: it is satisfied by both node
families, but it does not pin FMA contraction or BLAS kernel selection.  Any
future measurement campaign on this cluster should exclude `th-cl-dell0*`.

### The r<27 prior loses positive-definiteness

Separate from the retrain.  The 10,000,000-atom prior (uniform over truth
r<27 plus every truth r<20 source row) assembled cleanly — 10,237,320 atoms,
the median floor binding on exactly 50.0% of atoms in all four target
coordinates, rebuild verified to reproduce the assembled proposal exactly
(job 16637779).  Two 10,000-observation windows then ran against it (jobs
16637857, 16637858, ~33 min each).

| | bright20 w1 | bright20 w2 | f27 w1 | f27 w2 |
|---|---|---|---|---|
| eigenvalues | +717,035 / +1,789,237 | +201,607 / +2,420,665 | **−846,701 / +2,765,907** | +201,854 / +2,518,215 |
| positive definite | yes | yes | **no** | yes |
| bootstrap positive (400x) | 89.0% | 68.0% | **18.5%** | 61.5% |
| Pareto k median / >0.7 | 0.557 / 31.31% | 0.568 / 31.35% | 0.546 / 25.19% | 0.552 / 25.60% |
| worst row, share of \|total g1\| | 30.76% | 104.17% | 66.46% | **386.23%** |

Pooled over all 20,000 observations the r<27 prior is still non-positive
(−566,425 / +5,205,701) and yields no estimate; the unbounded prior pools to
+935,035 / +4,193,509 and does.  The bright20 columns reproduce the
2026-09-22 entry below exactly, which is what validates this pipeline.

The r<27 prior's *average* importance-sampling health is better (25.2% of rows
above Pareto k 0.7 against 31.3%).  It fails on the extremes, not the average.

**Confound, not yet separated:** the r<27 prior has 10,237,320 atoms against
the unbounded prior's 20,231,221, so "the bound broke it" and "half the atoms
broke it" are not distinguished.  The better Pareto k argues against the atom
count, but rebuilding the r<27 prior at 20M atoms is what would settle it.

### These windows cannot verify a 0.3% bias, and never could

Recorded because it was not stated before the windows were run.  Centred at
the run's own `initial_center` and against the injected g1 = +0.02:

| arm | g1 | m |
|---|---|---|
| bright20 w1 | +0.03482 ± 0.01471 | +74.1% ± 73.5% |
| bright20 w2 | +0.05118 ± 0.08591 | +155.9% ± 429.5% |
| bright20 pooled 20,000 | +0.03769 ± 0.01724 | **+88.4% ± 86.2%** |
| f27 w2 | +0.09735 ± 0.39628 | +386.7% ± 1981.4% |

The best arm's uncertainty is ~290x the 0.3% target.  The full
500,000-observation production run fails the same way — 41.96% of its rows
carry negative g1 curvature and its net information is −2.12% of its positive
mass — which is the combiner failure already recorded on 2026-09-20.  The
0.3% question therefore has to be answered through the m measurement path,
not through this one.

### Limitations

- The flow is not retrained.  Only its training measurement is ready.
- The r<26.5 emulator has not been differenced against the r<26 one on a
  common evaluation set; only its training metrics are reported.
- The 10M-atom r<27 prior and the 20M-atom unbounded prior differ in two ways
  at once, as noted above.
- Cases 77 and 85 keep a value that differs from the 20260917 store.  The
  evidence that the stored value is the defective one is strong but indirect:
  the old campaign records no hostname, so the attribution rests on the sacct
  node map for array 16556786.
- `m` from the 80-case path was −0.200 ± 0.264 pp; that error bar equals the
  target, so 80 cases can only bound |m| <= 0.3%, not resolve it.  Resolving
  to ±0.10 pp needs roughly 560 cases.

### Next steps

1. Restore the 17-module archived flow-training closure into an **isolated
   checkout**, not the live worktree.  Restoring it into the worktree adds
   files to `sbsi/`, which `run_disk_inference.py implementation()` hashes,
   and that invalidated 20 prepared shards earlier today.
2. `prepare_full_domain_flow.py --truth-magnitude-max 26.5` over the
   re-measured parent, then the refinement chain, then retrain.
3. Measure m on both the old r<26 evaluation sample (non-inferiority, isolating
   the model change) and the new r<26.5 sample (the real target).
   `doc/V36_INFERENCE_REVIEW.md` §1: do not change the population to pass the
   gate.
4. Optional, owner's call: rebuild the r<27 prior at 20M atoms to separate the
   bound from the atom count.

## 2026-09-22 — MAG_ERR near 26 is 0.13, but the real truth-to-measured scatter is 0.39; measured directly, r<26.5 captures 97.2% of the selected sample

Owner asked for the mean/typical `MAG_ERR` of objects near `MAG_AUTO` 26, the
number behind the proposed 26.5 training domain.  Answered from the rendered
simulation itself, with no model in the loop: SExtractor catalogues,
crossmatch and generated truth for constant-shear cases 100-109, plus leg.
`magerr.py`, job-local under `$CLAUDE_JOB_DIR/tmp`, job 16637408 (CPU).

### Files and behaviour changed

None.  This is a measurement, and it corrects the model-based bracket in the
entry below.

### The reported photometric error

Median `MAGERR_AUTO` by measured `MAG_AUTO`, mean +- s.e. over 10 cases:

| MAG_AUTO | MAGERR_AUTO |
|---|---:|
| 25.0-25.5 | 0.0843+-0.0000 |
| 25.5-26.0 | 0.1137+-0.0000 |
| 26.0-26.5 | 0.1467+-0.0000 |
| 26.5-27.0 | 0.1804+-0.0000 |

At `MAG_AUTO` 26.0 the typical reported error is about **0.13 mag**.  Taken
literally, "26 plus a typical `MAG_ERR` near 26" lands at 26.13, not 26.5.

### But the reported error is not the scatter that matters

`MAGERR_AUTO` is the photometric noise estimate.  What decides how faint a
truth domain must reach is the full truth-to-measured displacement, which also
carries blending.  Measured minus true magnitude, by true r, over the same 10
cases:

| true r | median shift | robust sigma |
|---|---:|---:|
| 25.0-25.5 | -0.1517+-0.0005 | 0.2713+-0.0007 |
| 25.5-26.0 | -0.1518+-0.0010 | 0.3249+-0.0007 |
| 26.0-26.5 | -0.0886+-0.0005 | **0.3919+-0.0009** |
| 26.5-27.0 | -0.0006+-0.0007 | 0.4658+-0.0007 |

Near true r 26 the real scatter is **0.39 mag, about 2.7 times the reported
0.147**, and the median measurement is **brighter** than the truth by roughly
0.09-0.15 mag.  Blending adds flux; that systematic brightening plus the wider
scatter, not photometric noise, is what carries faint objects across
`MAG_AUTO < 25.8`.

So 26.5 is a defensible boundary, but not for the stated reason.  It is about
26 plus 1.3 robust sigma of the real displacement, not 26 plus a typical
`MAG_ERR`.

### What each boundary costs, measured rather than modelled

True-magnitude composition of the sample that actually passes
`FLUX_RADIUS > 3.0` px and `MAG_AUTO < 25.8`, mean +- s.e. over 10 cases:

| true r | share of the selected sample |
|---|---:|
| r < 24.0 | 22.341+-0.013 |
| 24.0-25.0 | 27.402+-0.027 |
| 25.0-25.5 | 20.264+-0.026 |
| 25.5-26.0 | 19.119+-0.026 |
| 26.0-26.5 | 8.066+-0.021 |
| 26.5-27.0 | 2.134+-0.009 |
| 27.0-27.5 | 0.505+-0.006 |
| r >= 27.5 | 0.168+-0.002 |

| prior cut at | misses, % of the selected sample |
|---|---:|
| r < 25.0 | 50.257 |
| r < 26.0 | **10.873** |
| r < 26.5 | **2.808** |
| r < 27.0 | **0.674** |

This reproduces the recorded identity join over the frozen 500,000
observations, 54,231 at true r>=26 or 10.8462%, to within 0.03 points, from an
independent path.  The two agree, so the rest of the table is trustworthy.

### The model-based bracket in the entry below was 2-4x too pessimistic

| cut at | model bracket (entry below) | measured here |
|---|---|---:|
| r < 26.0 | 17.8 to 22.9% | **10.873%** |
| r < 26.5 | 7.5 to 10.5% | **2.808%** |
| r < 27.0 | 2.6 to 4.0% | **0.674%** |

The direction was right and the reason stands: the prior and likelihood
together manufacture faint-truth mass beyond the training domain.  The size is
now pinned from the simulation rather than bracketed from the model, and the
manufactured excess is larger than first reported, a factor 1.6-2.1 at r>=26
and 2.7-3.7 at r>=26.5.  Use the measured column; the bracket is superseded.

### The r<26 limit is a selection, not a generation limit

Each case generates **699,568** galaxies, of which **31.72%** lie at true
r<26 -- identical to the 31.721% census of the 139,936,000-row source prior
catalogue, confirming the prior source is the generated population.  The
V3.6 flow's truth parent is about 110,695 rows per case, close to half of the
221,921 generated at r<26, consistent with the secondary-role split at
`input_index < floor(N/2)`.

The renders therefore already contain objects to r~29; `r<26` is applied when
target rows are chosen, not when images are made.  **Extending the training
truth domain to 26.5 is a re-selection of existing renders plus a retrain, not
a new simulation campaign.**  Confirmation still belongs to the BlendEMU side,
but the counts leave little room for another reading.

### Limitations

- Ten cases of the constant-shear set, plus leg only.  Case-to-case spread is
  tiny, but a systematic shared by all cases would not show up.
- Robust sigma is the interquartile range scaled by 0.7413; the displacement
  distribution has heavy tails, so it understates the extremes that actually
  carry the faintest objects across the cut.
- Objects generated but never detected have no measured magnitude and are
  absent from the displacement table by construction.
- Extending the domain to 26.5 requires re-establishing the ~0.3% result on
  the wider parent; it does not carry over from the r<26 evidence.

### Next steps

1. Finish the r<27 prior (job 16637206, preparing): assemble, rebuild the
   median-floored proposal, run the two 10,000-observation windows.
2. An r<26.5 prior now looks like the right long-run target: it misses 2.8% of
   the selected sample and would sit exactly on the proposed training domain.
3. Ask the BlendEMU side to confirm the target-row selection can be widened to
   26.5 without re-rendering.
## 2026-09-22 — Priced the three population boundaries; under the real cuts r<26 costs 18-23%, and the model claims twice as much faint truth as the catalogue holds

Owner set out the boundary contract: the simulation should hold almost the
whole catalogue; the likelihood should reach about 0.3% under the realistic
measured cuts while being trained on a wider truth domain than those cuts
imply; the prior should cover as much mass as possible without leaving the
training domain.  This entry checks where each boundary actually sits and
measures what moving the last one costs.  No new GPU work; `selmass.py`,
job-local under `$CLAUDE_JOB_DIR/tmp`, job 16637313 (CPU).

### Files and behaviour changed

None.  This is a measurement and a correction to a stated premise.

### Where the boundaries actually sit

| boundary | contract | today |
|---|---|---|
| simulation | almost the whole catalogue | renders the whole scene; the source prior catalogue spans r 11.52-29.00 |
| likelihood, truth domain | wider than the measured cuts, to be extended | **true r<26**, inherited from the simulator's target-role parent (`CONVENTIONS.md` "Fixed-g0 retraining domain") |
| likelihood, measured cuts | 0.6 arcsec and 25.8 | `FLUX_RADIUS > 3.0` px and `MAG_AUTO < 25.8`, unchanged |
| prior | as much mass as possible, inside the training domain | uncut to r=29, i.e. **three magnitudes outside** the training domain |

The truth training domain is **r<26, not 25.8**.  25.8 is the measured
`MAG_AUTO` cut and is correctly a measured cut, not a truth cut; the two are
separate boundaries that happen to sit near each other.  The extension the
owner has in mind is therefore +0.5 mag from 26, not +0.7 from 25.8.

### What a truth boundary costs under the realistic cuts

Earlier entries priced truth cuts against *detected* mass only, because the
per-atom measured-selection probability is not cached.  It can be bracketed
from what `prepare` already stored for the 20.2m prior: per-atom flow mean and
scatter in the four measured coordinates, per-atom detection probability, and
per-atom prior weight.  Treating the radius and flux cuts as independent
bounds the joint pass probability from below; treating them as perfectly
dependent (`min` of the two tails) bounds it from above.

The bracket is checked, not assumed.  Against the exact 64-sample Monte-Carlo
`selected_mass` the same preparation stage wrote into each shard manifest:

```
CHECK exact selected mass  = 0.267788431
CHECK bracket              = [0.247981217, 0.285121594]
CHECK exact inside bracket = True
CHECK exact sits at        = 0.533 of the way up
```

Containment holds in all 20 shards individually, and the exact value sits near
the middle of the bracket in every one.  Nothing was tuned to make them agree.

Share of each quantity by truth magnitude, mean +- s.e. over the 20 shards:

| truth bin | % atoms | % detected | % selected (indep) | % selected (dep) |
|---|---:|---:|---:|---:|
| r < 24.0 | 7.377+-0.006 | 11.390+-0.010 | 22.713+-0.017 | 19.791+-0.016 |
| 24.0-25.0 | 8.731+-0.005 | 15.224+-0.011 | 26.008+-0.016 | 23.078+-0.015 |
| 25.0-25.5 | 7.174+-0.005 | 11.918+-0.007 | 17.659+-0.009 | 16.746+-0.008 |
| 25.5-26.0 | 9.224+-0.005 | 14.342+-0.010 | 15.864+-0.011 | 17.439+-0.011 |
| 26.0-26.5 | 12.328+-0.007 | 17.283+-0.010 | 10.271+-0.008 | 12.437+-0.009 |
| 26.5-27.0 | 15.767+-0.009 | 17.499+-0.013 | 4.888+-0.006 | 6.491+-0.008 |
| 27.0-27.5 | 17.781+-0.008 | 10.415+-0.007 | 2.067+-0.002 | 3.149+-0.004 |
| 27.5-28.0 | 13.390+-0.007 | 1.829+-0.002 | 0.485+-0.001 | 0.796+-0.001 |
| r >= 28.0 | 8.229+-0.006 | 0.101+-0.000 | 0.046+-0.000 | 0.072+-0.000 |

Mass a truth cut would lose, as a percentage of the measured-cut population:

| cut at | % detected lost | % selected lost |
|---|---:|---:|
| r < 25.0 | 73.386 | 51.3 to 57.1 |
| r < 26.0 | 47.127 | **17.8 to 22.9** |
| r < 26.5 | 29.844 | **7.5 to 10.5** |
| r < 27.0 | 12.345 | **2.6 to 4.0** |
| r < 27.5 | 1.930 | 0.53 to 0.87 |

The measured cuts do most of the work a truth cut would do.  Cutting at the
current training boundary r<26 costs 18-23% of the population that survives
selection, not the 47% the detection-only accounting suggested.  Extending the
training domain to 26.5 takes the prior from representing about 80% of the
selected population to about 91%; extending to 27 takes it to about 97%.  The
owner's own heuristic, 26 plus a typical `MAG_ERR` near 26, lands at 26.5,
where the curve has already flattened by more than half.

### The model claims about twice as much faint truth as the catalogue has

The same table predicts what fraction of the *observed* sample should be at
true r>=26: 17.8-22.9%.  The completed identity join over the frozen 500,000
observations, recorded in `doc/V36_INFERENCE_REVIEW.md`, finds 54,231
(10.8462%) at true r>=26 with zero unmatched rows, consistent with the 10.89%
eligible-population fraction.

So the prior and likelihood together put **1.6 to 2.1 times more faint-truth
mass into the selected sample than the catalogue actually contains.**  Both
the detection classifier and the flow are extrapolating beyond r<26 there, and
this is the same region where the flow already produces greater-than-5-mag
bright-side errors on 6.9% of prior atoms.  The excess is therefore at least
partly manufactured mass, which makes the case for bounding the prior at the
training domain stronger than the loss table alone: some of the 18-23% that a
r<26 prior would discard was never real.

This comparison does not separate the classifier's contribution from the
flow's; that needs the per-atom selection probability recomputed at truth-bin
resolution, which is a small GPU job and was not run.

### Limitations

- The per-atom selection probability is bracketed, not exact.  The bracket is
  validated on the total but its *shape* across truth bins is not separately
  validated, so the bin-by-bin numbers carry the bracket width and no more.
- `probability.npy` is read at the zero-shear node while `values.npy` and
  `dispersion.npy` are the proposal-coordinate summaries at the centre node.
  The offset is far smaller than the bracket.
- The 10.85% observed fraction comes from the recorded identity join, not
  recomputed here; the input `truth.parquet` carries identity keys only.
- Whether the r<26 parent can be widened by re-selecting existing renders or
  needs new simulation is not settled here.  The per-case counts in
  `initial_main_truth_parent.json` (about 110.7k truth-parent rows against
  about 313.5k crossmatched rows) are consistent with the renders already
  containing fainter objects, but that is an inference from counts and belongs
  to the BlendEMU side to confirm.

### Next steps

1. The r<27 prior the owner asked for is preparing (job 16637206, 8/20 shards
   complete, no errors).  Finish it: assemble, rebuild the median-floored
   proposal, and run the same two 10,000-observation windows.
2. Offer an r<26 arm.  It is the only prior that respects the current training
   domain exactly, and uncut/r<27/r<26 together measure how much the estimate
   is driven by extrapolation rather than by population.
3. Extending the training domain to r<26.5 requires re-establishing the 0.3%
   result on the wider parent; it does not carry over from the r<26 evidence.
## 2026-09-22 — A truth cut narrows the population, so it gets its own format; r<27 discards 39.9% of the catalogue but 12.3% of what contributes

Owner asked for a prior of 10,000,000 uniform atoms drawn only from truth
r<27, plus every source row at r<20.  Unlike the bright stratum, this is a
genuine change to the represented population and is recorded as one.

### Files and behaviour changed

- `scripts/subsample_disk_prior.py`: `bright_source_rows` generalized to
  `source_rows_below(..., cuts=...)`, which returns the rows below each of
  several cuts from **one** pass over the truth column, so the bright stratum
  and the eligible frame cost a single read between them.  `stratified_rows`
  gained an optional `eligible` frame: the uniform draw is taken inside it and
  the inclusion probability is `size/len(eligible)`, not `size/total`.  New
  `--faint-cut`; `--bright-column` now names the column both cuts read, and
  `prepare` refuses `bright_cut >= faint_cut`.
- `sbsi/disk_inference_store.py`: added
  `truth_cut_disk_prior_subset_v3`/`truth_frame_uniform_plus_certain_stratum`
  to `SUBSET_SAMPLING` and exported `TRUTH_CUT_SUBSET`.  The manifest guard
  now requires `truth_cuts` to be **non-null for v3 and null for v1/v2**, so
  a cut subset cannot be read as the uncut population and an uncut subset
  cannot claim a cut it does not have.  Message is now "invalid complete
  subset manifest".
- `tests/test_stratified_disk_prior.py`: +7 tests (19 total) covering the
  frame-restricted draw, frame-relative weights, the refused inverted cut
  pair, an undeclared cut, a falsely declared cut, and an out-of-range frame.

`subset_weights`, `load_source_shard`'s pilot guard and the `run` stage needed
no change: they already branch on `prior_weight is None`.  The four pinned
preparation functions are untouched.

### What the cut costs, measured rather than assumed

Truth `r` census over all 139,936,000 source rows, joined to the 20.2m prior's
per-atom weight and the per-atom detection probability `prepare` already
stored (jobs 16636034 and 16636395, CPU).  "Detected mass" is
`sum(weight * detection_probability)` at the centre node — what
`detected_selected_mass_shard` integrates, without the measured
flux-radius/MAG_AUTO factor, which is not stored per atom.

| cut | source rows | % atoms | % prior weight | % detected mass | **mass a cut here loses** |
|---|---|---|---|---|---|
| r<24 | 8,828,485 | 7.38% | 6.31% | 11.39% | 88.61% |
| r<25 | 21,183,426 | 16.11% | 15.14% | 26.61% | 73.39% |
| r<26 | 44,388,838 | 32.51% | 31.73% | 52.87% | **47.13%** |
| r<26.5 | 61,833,660 | 44.83% | 44.20% | 70.16% | 29.84% |
| **r<27** | **84,156,318** | **60.60%** | **60.15%** | **87.66%** | **12.35%** |
| r<27.5 | 109,335,187 | 78.38% | 78.13% | 98.07% | 1.93% |

The r<26 row is a check, not a new number: the 2026-09-21 population audit
(job 16617937) independently reported r>=26 atoms carrying "approximately
47.14%" of usability-weighted mass, against 47.13% here.  The two agree, so
the decomposition is trustworthy at the other cuts.

r<27 therefore discards **39.86% of the catalogue by count but 12.35% of the
detected mass**: the discarded objects are the ones that are almost never
detected.  This is also the region where the retained model evidence stops —
`doc/V36_INFERENCE_REVIEW.md` records the support gap at true r>=26 — and
where the flow's own predictions are least credible (95.9% of atoms it calls
measured-bright have truth r near 28).

### The built subset

```
SUBSET_COMPLETE atoms=10237320
format       truth_cut_disk_prior_subset_v3
truth_cuts   r < 27.0; frame 84,156,318 of 139,936,000 (60.139%); 55,779,682 discarded
bright       r < 20.0; 269,501 source rows, 237,320 beyond the uniform draw
uniform p    0.11882649143466567     weight ratio 8.4157
```

Bright atoms are 2.632% of the subset and carry 0.3202% of its weight; their
true share **of the frame** is 0.3203%.  The weights restore the frame, which
is now what the prior represents.  Enrichment rises from 7.00x to 8.42x
because the uniform frame shrank, and the subset is half the size of the
20.2m build, so preparation should cost about 1.1 GPU-hours.

### Limitations

- **The population is no longer the source.** Rows at r>=27 have inclusion
  probability zero and no weight can restore them, so every number from this
  prior is conditional on r<27 while the 500,000 observations still contain
  those objects.  This is a deliberate diagnostic, not a step toward the
  documented uncut inference, and `doc/V36_INFERENCE_REVIEW.md` §1 explicitly
  forbids changing the population to pass the domain gate.
- Changing `sbsi/disk_inference_store.py` again retires the 20.2m prepared
  cache under `check_prepared_identity`, exactly as the 24m cache was retired.
  Both remain valid on disk and re-runnable from commit `e2f8b9e`; the
  comparisons below read their stored moments instead.
- The mass decomposition omits the measured flux-radius/MAG_AUTO factor.
- A fresh seed (20260922) was used, so the frame and the draw both differ from
  the 20.2m arm; the two are not a controlled single-variable pair.

### Next steps

1. Prepare (job 16637206), assemble, rebuild the median-floored proposal, and
   run the same two 10,000-observation windows for comparison against the
   stored 20.2m moments.
2. Unchanged and still the binding constraint: the weight tails.  The draw
   budget is not — see the entry below.

## 2026-09-22 — 16,384 draws is the right budget: the loss is the weights, not the sample size

Owner asked whether the draw budget should change.  Answered from the three
completed runs, no new compute (`draws.py`, job-local under
`$CLAUDE_JOB_DIR/tmp`).

### Files and behaviour changed

None.  This entry records a measurement and a decision not to change a setting.

### The budget is spent, not wasted, and then discarded by the weights

| | 20.2m w1 | 20.2m w2 | 24m uniform |
|---|---|---|---|
| distinct atoms drawn, median | 16,334 | 16,334 | 16,343 |
| as a share of the 16,384 budget | 99.7% | 99.7% | 99.7% |
| ESS 10th / median / 90th | 19 / 385 / 2,536 | 19 / 372 / 2,345 | 20 / 366 / 2,513 |
| median ESS as a share of budget | 2.35% | 2.27% | 2.23% |
| 1st-percentile ESS | 2 | 2 | 3 |
| rows with ESS < 100 | 27.70% | 27.90% | 28.27% |
| Pareto k >0.5 / >0.7 / >1.0 | 58.6 / 31.3 / 11.6% | 60.0 / 31.4 / 11.2% | 59.2 / 31.9 / 11.6% |

Duplication is not the problem: 99.7% of the 16,384 draws are distinct atoms.
The budget is then thrown away by the weighting — the median row retains 2.3%
of it.

### The loss is attributable to the weight tails

Rank correlation between ESS and Pareto k is **-0.82** in all three runs.
Splitting on the k=0.5 threshold, above which the importance-sampling variance
is no longer finite and extra draws stop paying at the usual rate:

| | share of rows | median ESS |
|---|---|---|
| k < 0.5 (well behaved) | 41.4% | **1,211** |
| k > 0.7 (heavy tailed) | 31.3% | **38** |

A well-behaved row already converts the budget 32x better than a heavy-tailed
one.  Doubling to 32,768 draws would buy about sqrt(2) in noise on the 41% that
behave and close to nothing on the 31% that do not, for twice the GPU.  Making
every row behave like the k<0.5 population would raise the median ESS by about
3.1x at no extra cost.

### Decision

Keep 16,384.  It is not the binding constraint in either direction, and it
should not be revisited until the weight tails are fixed, at which point the
same budget will deliver several times the effective sample.

### Limitations

- ESS and Pareto k are the final-rung diagnostics; they summarize the weight
  distribution and are not a direct measure of estimator error.
- The comparison assumes the standard importance-sampling rate argument; it
  does not prove what a 32,768-draw run would return, which was not run.

### Next steps

Unchanged: remove the exact stratum from the mixture before drawing, then
re-measure ESS at the same 16,384 budget.

## 2026-09-22 — The source catalogue caps bright enrichment at 7x for any cut; r<20 is exhausted, but loosening the cut is nearly free

Asked whether more bright atoms are available.  Census of the truth `r` column
over all 20 source shards (job 16636034, CPU, `bright_census.py` job-local
under `$CLAUDE_JOB_DIR/tmp`).

### Files and behaviour changed

None.  This entry records a measurement.

### The source population

139,936,000 rows, `r` from 11.520 to 29.000, **median 26.70**; the 0.1th
percentile is 19.37.  This is an overwhelmingly faint catalogue and the bright
end is genuinely scarce, not merely under-sampled.

| cut | source rows | % of source | expected in a 20M uniform draw | extra atoms to take all |
|---|---|---|---|---|
| r<19.0 | 93,862 | 0.0671% | 13,415 | 80,447 |
| r<19.5 | 160,852 | 0.1149% | 22,989 | 137,863 |
| **r<20.0** | **269,501** | **0.1926%** | **38,518** | **230,983** |
| r<20.5 | 438,210 | 0.3132% | 62,630 | 375,580 |
| r<21.0 | 700,560 | 0.5006% | 100,126 | 600,434 |
| r<21.5 | 1,092,865 | 0.7810% | 156,195 | 936,670 |
| r<22.0 | 1,675,778 | 1.1975% | 239,506 | 1,436,272 |
| r<23.0 | 3,831,419 | 2.7380% | 547,596 | 3,283,823 |
| r<24.0 | 8,828,485 | 6.3089% | 1,261,789 | 7,566,696 |

The "extra atoms" column is the expectation `n(1-p)`, `p = 20e6/139936000 =
0.142922`; the realised r<20 draw took 231,221 against an expected 230,983.

### 7x is a hard ceiling, at every cut

Taking *every* source row in a stratum sets its inclusion probability to 1
against `p` for the rest, so the enrichment over a uniform draw is `1/p =
6.997` **regardless of where the cut is placed**.  The r<20 build already
achieves it.  The weight ratio bright:faint is likewise fixed at 6.997 at any
cut, so loosening the cut costs nothing in weight dynamic range — a useful
property, given that weight tails are the current binding constraint.

Going beyond 7x requires one of:

- a **smaller uniform base** (10M uniform + all bright gives 14x), which trades
  faint coverage for bright coverage within the same atom budget;
- a **larger source catalogue**, which is BlendEMU's scope, not SBSI's
  (`doc/API.md`).

### Cost is not the obstacle

Prepare ran 2.2 GPU-hours for 20,231,221 atoms.  Scaling by atom count:

| cut | subset atoms | prepare estimate |
|---|---|---|
| r<20.0 (built) | 20,231,221 | 2.20 GPU-h |
| r<21.0 | 20,600,434 | 2.24 GPU-h |
| r<22.0 | 21,436,272 | 2.33 GPU-h |
| r<23.0 | 23,283,823 | 2.53 GPU-h |
| r<24.0 | 27,566,696 | 3.00 GPU-h |

Reaching r<22 costs about 6% more prepare than the build already completed.

### Why this should not be spent yet

The scarcity argument that motivated the r<20 stratum weakens as the cut
loosens: at r<20 a uniform 20M draw supplies only 38,518 atoms, but at r<22 it
already supplies 239,506, so the same 7x boost addresses a far less acute
starvation.  More importantly the entries above measured what the first 7x
boost bought: the summed information turned positive definite, but Pareto k was
unmoved (median 0.563 to 0.557, 31% of rows above 0.7) and the robust error
stayed 12–17x the model error.  Prior composition is no longer the binding
constraint, and a second prior intervention would be read through the same
noisy estimator.

### Limitations

- The cut is on truth `r`, available only because the source catalogue carries
  it.  This remains a diagnostic instrument, not a shippable prior
  construction.
- Prepare cost is extrapolated linearly in atom count from a single measured
  point; shard-level overhead is not separated out.

### Next steps

1. Unchanged: fix the importance-sampling weight tails (remove the exact
   stratum from the mixture before drawing) before any further prior work.
2. If a further bright intervention is ever wanted, the cheap ordering is
   r<21 then r<22 — but only after (1) makes the result measurable.

## 2026-09-22 — Second window confirms positive-definiteness is a property of the prior, but it passes only marginally and the estimate is unusable

Rows 10000–19999, never previously run, under the same bright-stratified
prior and the same settings as job 16631250.  This was the top caveat of the
entry below: 86.57% of the first window's improvement came from one
observation, so the result needed a window that observation is not in.

### Files and behaviour changed

None.  This entry records a measurement.  `window2.py` is job-local under
`$CLAUDE_JOB_DIR/tmp`.

### It reproduces

| | window 1 (rows 0–9999) | window 2 (rows 10000–19999) |
|---|---|---|
| eigenvalues | +717,035 / +1,789,237 | **+201,607 / +2,420,665** |
| positive definite | yes | **yes** |
| bootstrap positive (400x) | 89.0% | **68.0%** |
| Pareto k median / >0.7 | 0.557 / 31.31% | 0.568 / 31.35% |
| ESS median | 385.0 | 371.7 |
| negative-curvature rows | 6,464 | 6,471 |
| per-row g1 10th / median / 90th | −102.6 / +2.2 / +87.6 | −94.9 / +2.2 / +99.4 |
| worst row, share of \|total g1\| | 27.10% | **104.17%** |

Positive-definiteness is a property of the prior, not of rows 0–9999.  The
per-row distributions are near-identical between windows (median +2.2 in
both), which is what a prior-level change should look like.

### It passes only just

Window 2's smallest eigenvalue is 201,607 against window 1's 717,035, its
bootstrap holds in 68.0% of resamples against 89.0%, and **its single worst
row (14796, at −275,666) exceeds the whole summed g1 information**.  Dropping
that one row moves the smallest eigenvalue from 201,607 to 451,187.  The same
few-rows-dominate structure the entry below identified is still the governing
behaviour; the bright stratum lowered the ceiling on how bad a single row can
be, it did not remove the dependence.

### The estimate does not constrain anything yet

```
window 1 : g1 = +0.020912 +/- 0.001181 (model) +/- 0.014705 (robust)
window 2 : g1 = +0.037267 +/- 0.002198 (model) +/- 0.085906 (robust)
pooled   : g1 = +0.023777 +/- 0.001028 (model) +/- 0.017242 (robust)
           g2 = +0.002258 +/- 0.000500 (model) +/- 0.003530 (robust)
```

against an injected `[0.02, 0.0]`.  The two windows agree with each other at
0.19 sigma in g1 and 0.46 sigma in g2, so nothing is inconsistent — but the
pooled g1 is **+18.9% of signal with a robust uncertainty of ±86.2%**.  The
robust error is 17x the model error in window 2 and 12x in window 1.  No
statement about bias can be made from this; it is a working estimator, not a
measurement.

### Limitations

- Two windows of the same 500,000-observation catalogue under one injected
  shear.  This tests reproducibility across observations, not across shear
  values or catalogues.
- The bootstrap resamples rows only, so it does not capture the Monte Carlo
  noise inside a row.  68% and 89% are upper bounds on confidence.
- Both windows use the same prior draw.  Whether a second stratified draw at
  a different seed also lands positive definite is untested.

### Next steps

1. Importance sampling is unambiguously the binding constraint now: Pareto k
   median 0.56–0.57 in both windows, 31% of rows above 0.7, robust error 12–17x
   the model error, and a single row able to exceed the summed information.
   The candidate v1.4 change — removing the exact stratum from the mixture
   before drawing — targets this directly and should come before any further
   prior work.
2. Do not spend more GPU on prior composition until the weight tails are
   fixed; the two interventions after it will be measured through the same
   noisy estimator.
3. Still open, unchanged: report the flow's >5 mag bright-side predictions
   (6.9% of prior atoms, flux sigma/|x| median 10.5) to the BlendEMU side.

## 2026-09-22 — The bright stratum makes the information positive definite and yields the first estimate; the obstruction was a handful of rows, not the catalogue

The 20,231,221-atom bright-stratified prior built above was prepared,
assembled and run over the same 10,000 observations as job 16629404.  The
combined information is positive definite for the first time and
`run_adaptive_section5` returned an estimate.  The cause is narrower than the
result looks.

### Files and behaviour changed

None.  This entry records a measurement.  Comparison scripts are job-local
under `$CLAUDE_JOB_DIR/tmp` (`compare_bright.py`, `robustness.py`).

### The controlled pair

Both arms: median-floored proposal, flux compared multiplicatively, identical
draw ladder, seeds, stencil and observations.  They differ **only** in the
prior.

| | 24m uniform (16629404) | 20.2m bright-stratified (16631250) |
|---|---|---|
| atoms | 24,000,000 | 20,231,221 |
| truth r<20 atoms | 45,975 | 269,501 |
| eigenvalues of summed information | −2,574,626 / +612,252 | **+717,035 / +1,789,237** |
| positive definite | no | **yes** |
| estimate | refused | **produced** |
| Pareto k median / >0.7 | 0.563 / 31.89% | 0.557 / 31.31% |
| ESS median | 365.8 | 385.0 |
| rows with any negative curvature | 6,133 | 6,464 |

```
20.2m BRIGHT : g1 = +0.020912 +/- 0.001181 (model) +/- 0.014705 (robust)
               g2 = -0.000611 +/- 0.000748 (model) +/- 0.004462 (robust)
```
against an injected `[0.02, 0.0]`.  In fractional terms g1 recovers
**+4.6% ± 73.5%** on the robust error, or ±5.9% on the model error.  **The
robust error is the one to quote**: it is 12x the model error, which is what a
heavy-tailed weight distribution does, and Pareto k barely moved.  This run
does not yet constrain the response ratio `m` of `doc/CONVENTIONS.md` §7; it
is a one-step estimate of the shear, and its honest reading is "consistent
with no bias, with an uncertainty three quarters the size of the signal".

### The obstruction was a few rows, and the bright stratum removed the worst

Per-row g1 information, both arms:

| percentile | 24m uniform | 20.2m bright |
|---|---|---|
| 0.1th | −19,317 | −38,579 |
| 10th | −106.1 | −102.6 |
| 50th | +2.1 | +2.2 |
| 90th | +89.6 | +87.6 |
| 99.9th | +31,050 | +48,699 |

**The bulk is unchanged.**  What changed is the extreme tail: the single worst
row went from −2,467,302 to −194,417, a factor of 12.7.  Row 3563 alone — the
first row the earlier census singled out — moved from **−2,467,302 to
+220,670** and accounts for **86.57%** of the entire change in summed g1
information; the top five rows account for 93.48%.  Its distinct-atom count
rose from 9,791 to 14,443 of 16,384.

Dropping the worst rows from the uniform arm confirms the diagnosis:

```
24m uniform, all rows      : [-2574626, +612252]
24m uniform, drop worst  1 : [ -187628, +946124]
24m uniform, drop worst  5 : [ +387704, +1330000]
```

One row carried almost the whole negative total, and five rows carried all of
it.  The 2026-09-14 reading of this as a broad curvature property of the
likelihood was wrong; it was an outlier problem.

### This is not fragile, but it is not fixed either

Bootstrapping the row set 400 times, the smallest eigenvalue of the summed
information is positive in **89.0%** of resamples (median +543,864), against
**12.2%** (median −2,273,360) for the uniform arm.  The new result does not
rest on one lucky row.  That bootstrap resamples rows only, so it captures
row-to-row variation and not the Monte Carlo noise inside a row; 89% is an
upper bound on confidence.

Three things say the underlying defect is still live:

- Negative-curvature rows went **up**, 6,133 to 6,464, and 2,048 rows flipped
  from positive to negative while 1,717 flipped the other way.  The per-row
  sign is churning, not settling.
- New catastrophic rows appeared where none were: row 5907 went from +7,183 to
  −194,417 and row 2874 from +251,674 to −67,443, with Pareto k 1.49 and 2.20.
  The mechanism — a few atoms dominating a row's weight — is unchanged; it has
  landed on different rows.
- Pareto k is essentially identical: median 0.563 to 0.557, and 31.3% of rows
  still exceed 0.7.  The importance sampling is as bad as it was.

The correct summary is that the bright stratum removed the outliers that were
poisoning the sum, not that it repaired the estimator.

### Validation

- prepare: 20/20 array tasks COMPLETED (job 16630114), 6m32s–6m55s each,
  2.2 GPU-hours, throttled `%2` to hold the two-GPU limit.
- The selected mass is the check that the weights restore the population.
  Summed over 20 shards it agrees with the 24m uniform run to **+0.025% on
  every one of the 10 stencil nodes** (0.26778843 against 0.26772120 at node
  0).  Two independent draws of the same population, one of them over-sampling
  its bright end sevenfold, land 2.5 parts in 10,000 apart.
- assemble (job 16631248, 3m28s): `ASSEMBLED atoms=20231221`, the proposal
  rebuild reported `VERIFIED rebuild reproduces .../disk_assembled20m/proposal
  exactly`, and the median floor binds on exactly 50.0% of atoms in all four
  coordinates.
- The run recorded `runtime_cache_compatibility: identical` — a fresh
  preparation under the current implementation, not a whitelisted exception.
- Inference 2,148 s (job 16631250) against 2,222 s for the uniform arm;
  20.2m atoms cost no more than 24m did.

### Limitations

- The estimate's robust error is 12x its model error.  Quoting the model error
  would overstate the precision by an order of magnitude.
- 86.57% of the improvement is one observation.  A different 10,000-row window
  could behave differently; this was rows 0–9999, the same window as every
  earlier arm, chosen for comparability and not resampled.
- The bright stratum is defined on truth `r`, which exists only because the
  source catalogue carries it.  A real survey has no such column, so this is a
  diagnostic instrument, not a shippable prior construction.
- The 24m prepared cache remains unrunnable without reverting the store change
  (recorded in the entry above); the uniform arm's results were read from disk.

### Next steps

1. The binding constraint is now importance sampling, not curvature: 31.3% of
   rows have Pareto k above 0.7 and the robust error is 12x the model error.
   The candidate v1.4 change already noted — removing the exact stratum from
   the mixture before drawing — targets exactly this.
2. Confirm on a second, disjoint 10,000-row window that positive-definiteness
   is a property of the prior and not of rows 0–9999.
3. Still open, unchanged: report the flow's >5 mag bright-side predictions
   (6.9% of prior atoms, flux sigma/|x| median 10.5) to the BlendEMU side.

## 2026-09-22 — A bright stratum can be over-sampled if its weight pays it back: 20,231,221-atom prior holds every truth-bright source row

The owner chose the feasible maximum after "20m + 5m bright" was shown
impossible: 20m uniform atoms plus every remaining source row with truth
r<20, with inverse-probability weights so the represented population is
unchanged.  The subset is built and verified; preparation is running.

### Files and behaviour changed

- `scripts/subsample_disk_prior.py`.  New `bright_source_rows` reads the truth
  magnitude column from each source shard before the draw.  New
  `stratified_rows` returns the uniform draw plus every bright row it missed,
  globally sorted and deduplicated, with each row's inclusion probability: one
  for a bright row, `size/total` otherwise.  Weights are the normalized
  inverses of those probabilities, written per shard to a new
  `prior_weight.npy` (hashed into the shard receipt) and into the existing
  `galaxies["prior_weight"]` column.  New `--bright-cut` and `--bright-column`
  flags; **leaving `--bright-cut` unset reproduces the previous uniform build
  exactly**, including its `uncut_disk_prior_subset_v1` manifest.
- `sbsi/disk_inference_store.py`.  `load_subset_manifest` accepts the new
  `uncut_disk_prior_subset_v2` format, which carries `prior_weight: null` and
  a `bright_stratum` block, and still refuses a v1 manifest whose scalar
  weight disagrees with its row count.  New `subset_weights(manifest, index)`
  returns the per-atom weights, whole-subset or per shard, and refuses a set
  that does not sum to one.  `load_source_shard` uses them instead of
  `1/n_rows`, and refuses to truncate a stratified shard to a pilot, because
  the assembler's pilot rescaling cannot account for the dropped weight.
  `FrozenDiskCache` takes an optional `weights=`; omitted, it is uniform as
  before.
- `scripts/run_disk_inference.py`.  `run` passes `subset_weights(subset)` to
  `FrozenDiskCache`, which feeds both the population normalization and the
  proposal's base weights.
- `tests/test_stratified_disk_prior.py` (new, 12 tests).

`prepare` and `assemble` needed **no source change** and their hash pin is
intact.  Both already sum per-atom weights globally — `prepare` through
`detected_selected_mass_shard(..., cache.prior.weights, ...)` and `assemble`
through `np.sum(masses, axis=0)`, whose pilot rescaling is a no-op at full
coverage — so correct weights arriving from the store are enough.

### The weights restore the population exactly

`scripts/subsample_disk_prior.py --size 20000000 --seed 20260920
--bright-cut 20.0` (job 16630045, 11 min 14 s, CPU):

```
SUBSET_COMPLETE atoms=20231221 output=.../prior_subset20m_bright20
```

| | |
|---|---|
| atoms | 20,231,221 = 20,000,000 uniform + 231,221 extra |
| truth r<20 source rows | 269,501 — **all present** (38,280 found by the uniform draw) |
| inclusion probability | 1.0 bright, 0.14292247884747314 otherwise |
| weights | 7.146038993487891e-09 bright, 4.9999405629636075e-08 otherwise |
| weight ratio | 6.9968 = 139,936,000 / 20,000,000, exactly as designed |
| weight sum | 1.000000000000006 |

The check that matters: the bright stratum is **1.3321% of the atoms and
carries 0.1926% of the prior mass**, against a true source fraction of
0.1926%.  Over-sampling by a factor of seven is paid back to four decimal
places.  This is Hájek normalization — the inverse-probability weights scaled
to sum to one — which is what the uniform 1/n weights already do by
construction, not an empirical correction.

Against the 24m uniform subset's 45,975 truth-bright atoms this is a factor
of **5.86** more intrinsically bright atoms, at 84% of the atom count.

### Validation

- `pytest tests` minus the six files broken at HEAD by `a010491`
  (`ConditionalMeanFlowRA`, untouched here): **307 passed in 50.07 s**
  (job 16630044), against 295 before.  The 12 new tests cover the uniform
  path's exact reproduction, every bright row surviving, no duplication when
  the uniform draw already found a bright row, per-shard weight alignment
  against `galaxies.parquet`, and four refusals.
- The subset build was gated on those tests with `--dependency=afterok`.
- The bright pre-pass found 13,247–13,524 rows per source shard, summing to
  269,501 — the same count the 2026-09-21 census measured independently.

### Limitations

- **The 24m prepared cache is no longer runnable without reverting this
  change.**  `sbsi/disk_inference_store.py` is not on the driver's
  `RUN_STAGE_IMPLEMENTATION` whitelist, so `check_prepared_identity` now sees
  a third changed file beyond the audited `{driver, density}` pair and
  refuses.  The cache is in fact still valid — `subset_weights` returns
  bit-identical `1/n_rows` weights for a v1 manifest — but proving that to the
  gate needs a deliberate audited exception, which was not taken unprompted.
  The baseline result `ten_k_floor50_16628938` is already on disk, so the
  comparison below does not need it re-run.
- A stratified subset cannot be truncated to a pilot; `load_source_shard`
  refuses rather than silently mis-normalizing.
- The bright pre-pass reads `galaxies.parquet` before the per-shard hash
  verification.  That verification still runs and still aborts the build, so a
  corrupted source cannot reach the output; it is only read twice.
- No result yet.  The two previous interventions on this axis — the median
  dispersion floor and the multiplicative flux metric — each moved the answer
  by 0.01–0.1%.  Nothing here predicts this one will do better; it tests a
  different cause (too few bright atoms) rather than a different metric over
  the same atoms.

### Next steps

1. `prepare` over 20 shards (job array 16630099, `inter`, throttled `%2` to
   hold the owner's two-GPU limit), then `assemble` and a median-floored
   proposal at `--fractional-floor measured_flux_from_mag_auto`, the same
   recipe as `proposal_floor50_v1`.
2. A 10,000-observation inference, compared against
   `ten_k_floor50_16628938` on the same six obstructing rows: negative-curvature
   row count, combined-information eigenvalues, Pareto k and ESS.
3. Still open, unchanged: report the flow's >5 mag bright-side predictions
   (6.9% of prior atoms, flux sigma/|x| median 10.5) to the BlendEMU side.

## 2026-09-21 — "20m + 5m bright" cannot be built: only 269,501 truth-bright rows exist, and the measured-bright population is a flow artefact

The owner reaffirmed the request to build a 20m + 5m (mag<20) prior.  Three
CPU censuses were run before building.  The request cannot be executed as
stated, and the reason changes what "bright" means here.

### Files and behaviour changed

None.  This entry records three censuses and a blocked request.  Scripts are
job-local under `$CLAUDE_JOB_DIR/tmp` (`truth_mag_census.py`,
`bright_meaning.py`, `spurious_check.py`).

### The request is short by a factor of nineteen

Truth magnitude `r` is carried per atom in each source shard's
`galaxies.parquet`, so a bright stratum can be selected before `prepare`.
Over all 139,936,000 source rows (job 16629968, no non-finite values):

| truth cut | source rows | share |
|---|---|---|
| r<19 | 93,862 | 0.067% |
| r<20 | **269,501** | 0.193% |
| r<21 | 700,560 | 0.501% |

Source `r` has median 26.696 and 1st percentile 21.788.  A uniform 20m draw
already consumes about 38,518 of the r<20 rows, leaving about 230,983.  The
request asks for 5,000,000.  They do not exist.

### The measured-bright population is not bright

The earlier census counted 1,058,628 subset atoms at *predicted measured*
mag<20 (4.41%) while truth r<20 is 0.19% of the source, a factor of 23.  The
two axes were joined atom by atom over all 24m (job 16629976):

- 1,015,316 of the 1,058,628 measured-bright atoms — **95.9%** — have truth
  r >= 20, with truth r quartiles 27.805 / 28.148 / 28.452 against a source
  maximum of 29.0.  They are the faintest objects in the catalogue.
- They are **not** blend-brightened: `nbr_flux_near` median 0.4868 against
  0.7806 for the rest, and `nbr_flux_max` is indistinguishable (1.5124 against
  1.4485).  Neighbour light does not explain an eight-magnitude offset.

Across the whole prior the truth-minus-predicted offset has median +0.487 mag,
which is ordinary, but a heavy tail: 3,175,825 atoms (13.2%) predicted more
than 2 mag bright, 1,651,407 (6.9%) more than 5 mag, 987,791 (4.1%) more than
8 mag; the 99th percentile is +11.248 mag (job 16629983).

### The flow flags them, and the proposal already discards them

Those atoms carry a predicted flux scatter of `sigma/|x|` median **10.53**
(90th 11.26) against **0.77** (90th 1.58) for atoms whose prediction sits
within a magnitude of truth.  The flow reports an uncertainty ten times its
own prediction, and the dispersion-weighted rerank divides by it.

The selected 1,024 candidates were reconstructed for six obstructing rows and
four faint controls.  In every case **0.0%** of the selected candidates have
truth r >= 26, and the candidates' median truth `r` tracks the observation:

| row | observed mag | candidate truth r | candidate measured mag |
|---|---|---|---|
| 3563 | 18.45 | 18.459 | 18.538 |
| 2874 | 17.40 | 17.510 | 17.543 |
| 4708 | 20.03 | 20.122 | 20.188 |
| 6189 | 19.55 | 19.569 | 19.638 |
| 1165 | 19.69 | 19.726 | 19.788 |
| 6067 | 20.81 | 21.063 | 21.088 |

Adding 5m measured-bright atoms would therefore add 5m atoms the sampler
already throws away, and would leave the prior roughly a quarter composed of
them.

### What is actually scarce

On the truth axis the subset holds 15,981 atoms at r<19 and 45,975 at r<20.  A
bright observation's 1,024-atom candidate set is drawn from about 16,000
available atoms — roughly **6%** of the entire truth-bright population at that
brightness — against about 0.005% for a faint observation.  That is a real
resolution limit and it is the defensible form of the owner's instinct.

Its ceiling is fixed by the source: taking every remaining r<20 row gives
about 269,501 against the 45,975 present, a **5.9x** increase, not 100x.

### Limitations

- The candidate reconstruction uses the Gaussian residual term of the
  production reranker without its log prior-mass term, as in the entry below.
- Truth `r` and predicted measured `mag_auto` are different quantities; a
  sub-magnitude offset is expected and observed (median +0.487).  The claim
  rests on the 5-to-13 magnitude tail, not on the bulk.
- Why the flow produces those predictions was not investigated.  It is an
  emulator/flow-quality question and falls on the BlendEMU side of the
  boundary in `doc/API.md`.
- A stratified prior is currently refused by the store:
  `sbsi/disk_inference_store.py:47` requires `prior_weight == 1/n_rows`, and
  `:68` and `:82` hardcode uniform weights.  Per-atom inverse-probability
  weights would have to be threaded through `prepare`, `assemble` and
  `FrozenDiskCache` before any stratified subset could be run.

### Next steps

- Decide whether to build the feasible version: 20m uniform plus every
  remaining truth-r<20 source row (about 231k), with inverse-probability
  weights so the represented population is unchanged.  Cost is about 2.6
  GPU-hours of `prepare` (20 shards at ~460 s) plus a 37-minute inference,
  inside the two-GPU limit, plus the weight plumbing above.
- Weigh that against the evidence: raising the dispersion floor (7-8x more
  distinct atoms) moved the information 0.01-0.05%, and repairing the
  neighbour metric moved it 0.1%.  A 5.9x bright pool is the same kind of
  intervention and the candidates it would refine already match truth
  magnitude to about 0.1 mag.
- Report the >5 mag flow predictions to the BlendEMU side regardless.  They do
  not currently corrupt the inference, because the flow's own scatter flags
  them, but 6.9% of the prior carrying a ten-fold relative uncertainty is a
  model-quality finding in its own right.
## 2026-09-21 — the neighbour metric now compares flux multiplicatively; it is a real repair and it does not move the curvature

Following the census entry below, the flux-dominated neighbour metric was
fixed and tested on the same 10,000 observations.  The defect it names is
real and is repaired.  It is **not** the cause of the negative curvature.

### Files and behaviour changed

- `sbsi/catalogue_sampling.py`: `ProposalCoordinateTable` gains a
  `fractional_targets` declaration.  A declared coordinate is compared through
  `asinh(x / scale)` — logarithmic once `|x|` exceeds the additive scale,
  additive below it, and defined at zero and for negatives, where a ratio is
  meaningless.  The transform reuses the stored `scale` as the changeover
  point and the existing `fractional_mask` helper, so no new array is stored;
  the metric centre and spread are re-derived from `values`.  A new
  `standardize()` method is now the single map used by both the KD-tree
  (`:682`) and the observation query (`:826`), which previously recomputed the
  expression inline.  `save`/`load` carry the declaration at manifest
  version 5; versions 2-4 load unchanged.
  **With no declaration the metric is bit-identical to the previous
  expression**, so every existing cache and completed run keeps its meaning.
- `scripts/run_disk_inference.py`: new repeatable `--proposal-fractional
  TARGET`, which re-declares the loaded table in memory via
  `dataclasses.replace`, so no cache rebuild is needed.  The declaration is
  written to `result.json` as `proposal_fractional_targets`.
- `tests/test_proposal_fractional_metric.py`: new, 10 tests.

This is run-stage code only.  `sbsi/disk_inference_store.py` constructs the
table but never reads `standardized`, so the 24m-atom prepared cache stays
valid; the identity gate's changed-set minus its run-stage whitelist remains
exactly the audited driver/density pair, and the run recorded
`audited_two_entry_response_lru_v1+run_stage:sbsi/catalogue_null.py,sbsi/catalogue_sampling.py`.

### Validation

`pytest tests -q` (excluding the six files broken at `a010491`): **295 passed**
(285 before, 10 new), job 16629385.

Real 24m-atom table, same job.  Declaring flux fractional leaves `values` and
`dispersion` identical and moves the flux metric centre/scale from
49.78/81.27 to 0.5795/0.7175.  Nearest-atom distance for the six obstructing
rows falls from 0.626-4.668 to 0.132-0.832; all six gain 4-144 atoms within
one unit where five previously had none.  Faint controls barely move
(row 2743 0.139 -> 0.129; row 8499 0.301 -> 0.301), which is the intent: the
change bites only where the additive scale was wrong.

10,000 observations, job 16629404, identical to job 16628938 in proposal,
ladder and seeds and differing only in the metric.  2222 s against 2229 s, so
the change is free.

The prefilter defect is real (job 16629712).  The production path takes the
top 131,072 by metric distance and reranks those to 1,024 by the
dispersion-weighted score.  The two reranked sets overlap only 21-48%, and the
globally best-scoring atom was **outside** the additive net for five of the
six rows and inside the fractional net for all six.

The inference answer nevertheless does not move:

| row | additive information diagonal | fractional |
|---|---|---|
| 3563 | -2408106, -239622 | -2467302, -253568 |
| 4708 | -477331, -60001 | -476752, -59928 |
| 6189 | 53376, -403519 | 53361, -403512 |
| 2874 | 251862, 75517 | 251674, 75513 |
| 1165 | -222966, -39695 | -223055, -39702 |
| 6067 | 7998, -2628 | 7932, -2632 |

Rows carrying a negative curvature diagonal: 6121 -> 6133 of 10,000.  Summed
information eigenvalues -2502572/+449406 -> -2574626/+612252; not
positive-definite either way, so neither arm yields an estimate.  Pareto k
(median 0.564 -> 0.563, >1 at 11.70% -> 11.57%), ESS (median 364.4 -> 365.8)
and distinct-atom counts are unchanged.

The reason the repair does not propagate is visible in the same job: the best
score inside the additive net is -8.24 against -8.20 in the fractional net for
row 3563, and -6.77 against -6.74 for row 4708.  The additive net was missing
the champion atom while holding atoms within about 4% of it in weight.  A
better-shaped net therefore buys a better candidate list and an
indistinguishable likelihood.

### Limitations

- The overlap and champion figures use the Gaussian residual term of the
  production reranker without its log prior-mass term, which needs the frozen
  cache.  That term varies far less across candidates than the residual term,
  but the figures are an approximation of the production ranking, not a replay
  of it.  A `RuntimeWarning: overflow encountered in square` is raised by
  atoms whose tiny dispersion makes the residual enormous; those atoms score
  far below the maximum and do not affect the reported statistics.
- The metric is left **opt-in**.  It is more correct and costs nothing, but it
  changes no measured quantity, so promoting it to the `v1.3-infer` default
  would alter a release identity for no demonstrated benefit.  That is the
  owner's call.
- Six rows are not a sample of 10,000, and only flux was declared fractional.
  `measured_flux_radius` spans a wide range too and was not tested.

### Next steps

- Stop looking for the curvature obstruction in the proposal.  Two independent
  probes now point away from it: raising the dispersion floor gave the worst
  rows 7-8x more distinct atoms and moved their information by 0.01-0.05%
  (entry below), and a metric that recovers the globally best atom moves it by
  about 0.1% (this entry).  What is proposed is not what is wrong.
- Look instead at the target the score and information are built from: the
  `h = 0.001` finite-difference stencil for the 2x2 information, and the
  disk-likelihood response at these bright, large, elliptical objects.
- If the metric is wanted as the default, that needs a release-identity
  decision and a `doc/INFERENCE.md` §v1.3-infer amendment, not just the flag.
## 2026-09-21 — the bright end is not short of atoms; the neighbour metric is additive in flux

Owner asked how many prior atoms exist and proposed replacing the prior with
20m original atoms plus 5m extra brighter than mag 20.  Three census jobs were
run to test that.  The proposal is not supported, and the investigation located
a different cause for the obstructing rows.

### Files and behaviour changed

No library or driver code changed.  This entry records a diagnosis only.
Analysis scripts are job-local under `$CLAUDE_JOB_DIR/tmp`
(`bright_atoms.py`, `gap_axis.py`, `metric_check.py`).

### What was measured

Prior size: **24,000,000** atoms; atom magnitude median 25.76, 1st percentile
16.97, minimum 7.59.  Counts are exact censuses, so they carry no sampling
error.

The prior is over-supplied at the bright end relative to the first 10,000
observations, not starved (job 16629075):

| brighter than | atoms | share of prior | share of obs | atoms per obs |
|---|---|---|---|---|
| mag 18 | 440,325 | 1.83% | 0.07% | 26.2 |
| mag 19 | 718,119 | 2.99% | 0.28% | 10.7 |
| mag 20 | 1,058,628 | 4.41% | 0.70% | 6.3 |
| mag 22 | 1,742,794 | 7.26% | 4.53% | 1.6 |
| mag 24 | 3,925,941 | 16.36% | 26.64% | 0.61 |

Matching the observed mag<20 fraction would need ~168,000 atoms against the
1,058,628 present.  Adding 5m more would raise the over-supply to ~30x while
removing 4m atoms from mag>24, where the observations concentrate.

No coordinate is marginally starved for the six obstructing rows (job
16629254).  The nearest atom on each coordinate independently is 0.000
standardized units away, each row has 288k-500k atoms within a factor of two
in flux, and each observation lies inside the 0.1-99.9 percentile range of
those atoms on every coordinate.  A population difference does exist in the
joint density — at mag<20 the atoms are round and small (|e| median 0.040,
flux_radius median 3.94) while the observations are elliptical and large (|e|
median 0.261, flux_radius median 8.63) — but no observation exceeds the atoms'
99th percentile |e| (0.869), so this is a density mismatch, not absent support.

### Cause located

`ProposalCoordinateTable.standardized` (`sbsi/catalogue_sampling.py:231`)
divides every coordinate by one global robust scale, and
`CatalogueProposal.__init__` (`:682`) builds its KD-tree in that space;
`:837` queries it for `prefilter_candidates` before any reranking.  The
scales are g1 0.0237, g2 0.0238, flux_radius 1.056, **flux 81.27**.  The flux
scale is set by the faint bulk (prior median flux 49.8), so bright
observations sit 58-1344 standardized units out in flux against 1-22 units in
the other coordinates.  Flux therefore dominates candidate selection by about
two orders of magnitude.

The selected neighbours show this directly.  For row 3563 the nearest atom
matches flux to 0.08% (41706.14 against 41738.67) while missing g2 by 0.049,
which is 2.04 scale units.  The prefilter spends its resolution on a flux
match of no scientific value and lets the shear-carrying coordinate drift.
The heteroscedastic rerank cannot repair this because it only reorders the
pool the prefilter selected.

Comparing flux fractionally (log10) instead of additively gives the same six
rows 13-632 atoms within 1.0 unit at nearest distances 0.125-0.347, against
0.14-0.51 for typical faint rows that already converge (job 16629309).

This is the additive/multiplicative distinction already applied to the
dispersion floor in the 2026-09-21 cache rebuild
(`fractional_mask`, `measured_flux_from_mag_auto`).  It was applied to the
floor but not to the coordinate scale the neighbour search uses.

### Limitations

The link from a flux-dominated candidate pool to the negative curvature
reported for these rows is inferred, not demonstrated.  A stable but
shape-mismatched candidate set is consistent with the observed behaviour
(converged, definite, wrong sign) but no run has yet been made with a
fractional flux metric to confirm it.  Changing `standardized` alters
preparation-stage geometry and would invalidate the prepared identity, so a
test must either go through a fresh `prepare`/`assemble` or be scoped to the
run stage explicitly.

### Next steps

- Decide whether to compare flux in log space inside `standardized`, or to
  carry a per-coordinate fractional flag on the table as the floor already
  does.  Prefer reusing the existing `fractional` knob over adding a new one.
- Re-run the six obstructing rows under the chosen metric and check whether
  the combined information becomes positive-definite.
- Do not change the prior's atom count on the basis of the bright-atom
  hypothesis; the census does not support it.
## 2026-09-21 — 10,000 observations: the proposal collapse is fixed, the curvature obstruction is not, and it is not a sampling artifact

Owner asked whether to run 10,000 observations. Run as job 16628938: rows
0-9999 against the median-floored proposal table, `--compile-flow
--object-chunk 16` to match the completed 500k production run, whose
`part_00` supplies the old-proposal answer for exactly those rows. One GPU,
39 minutes, 0.223 s/row. Only the new arm needed a GPU.

### The hypothesis this run was built to test, and its result

The entry below diagnosed the pathological rows as proposal collapse: the
rows carrying enormous negative information were the rows where the proposal
reached only ~1,800 distinct atoms of 16,384 while a healthy row reached
15,950. The median floor was expected to fix those rows by fixing the
collapse.

**The collapse is fixed. The curvature obstruction is not.**

| rows 0-9999 | old proposal | median floor |
|---|---|---|
| distinct atoms, median | 15,950 (97.4%) | **16,343 (99.7%)** |
| distinct atoms, 10th pct | 8,434 | **16,020** |
| rows collapsed below 25% | 4.67% | **0.00%** |
| summed information, eigenvalues | −2.923e6, +4.309e5 | −2.503e6, +4.494e5 |
| positive definite | no | **no** |
| Pareto k, median | 0.469 | 0.564 |
| Pareto k > 1 | 2.42% | **11.70%** |

No row collapses any more — the proposal defect is gone, completely and as
designed. The summed information is 14% less negative and still indefinite,
so **neither arm yields a shear estimate**;
`combine_inference_partitions.py` refuses both with `non-positive combined
information`. There is no number to report and therefore no uncertainty to
attach to one.

### The extreme curvature is real, not a sampling artifact

This is the substantive finding, and it falsifies the diagnosis in the entry
below. Four of the five rows that dominated the old sum were given five to
eight times more distinct atoms and did not move:

| row | atoms | g1 information | change |
|---|---|---|---|
| 4708 | 1,771 → 14,177 | −477,102 → −477,331 | **0.05%** |
| 6189 | 1,796 → 12,688 | −403,470 → −403,519 | **0.01%** |
| 2874 | 1,850 → 6,974 | 252,293 → 251,862 | 0.17% |
| 3563 | 1,840 → 9,917 | −2,565,207 → −2,408,106 | 6.1% |
| 6067 | 1,886 → 15,341 | −299,413 → **+7,998** | sign flip |

Drawing an eight-fold more diverse set of atoms and recovering the same
number to four significant figures is the signature of a *converged*
estimate, not a broken one. For those rows the old proposal, collapsed as it
was, was already returning the right answer. Their huge negative curvature is
a property of the likelihood at those objects.

Row 6067 is the exception that proves the rule: it was genuinely a sampling
artifact and the floor removed it entirely, from −299,413 to +7,998, which is
back to the healthy scale (the median row carries about +3).

The collapse and the bad curvature were correlated but not causally linked in
the direction assumed. Both are consequences of the same objects being
extreme, not of one another.

### What the obstruction actually is: two objects

Dropping rows by largest `|information|` and re-summing:

```
            old arm                        new arm
drop  0   [-2923236,  430851]            [-2502572,  449406]
drop  1   [ -400281,  754239]            [ -216461,  811022]
drop  2   [    5146,  885888]  POS DEF   [  116357, 1015537]  POS DEF
drop  5   [  265592,  934880]  POS DEF   [  501947,  915371]  POS DEF
drop 25   [  222634,  392882]  POS DEF   [   46762,   80909]  POS DEF
drop 50   [    5675,  175549]  POS DEF   [  -94810,   39400]
```

**Two rows out of ten thousand are the entire obstruction, in both arms.**
This was never a diffuse sampling problem needing a better sampler.

The sum also degrades once too many rows are dropped, because the healthy
bulk carries very little: 10,000 rows at a median of ~3-5 is about 4e4 of
information, while single outlier rows carry 1e5 to 1e6. The extremes are ten
to a hundred times the entire bulk. That imbalance is the real structural
problem.

### What those objects are

The six largest-`|information|` rows in the new arm are the brightest,
largest and most elliptical objects in the sample:

| row | mag_auto | flux | flux_radius | measured g |
|---|---|---|---|---|
| 3563 | 18.45 | 41,706 | 6.54 | (−0.001, 0.161) |
| 4708 | 20.03 | 9,709 | 6.90 | (−0.061, 0.308) |
| 6189 | 19.55 | 15,205 | 6.88 | (−0.506, −0.085) |
| median of the 10,000 | 24.71 | 131 | 4.18 | (0.010, −0.001) |

They are 4.7 to 6.3 magnitudes brighter than the median object, carry 74 to
318 times its flux, and reach ellipticities up to 0.51. This is the same
population that the 2026-07-05 bright-neighbour work identified as outside
the emulator's training domain, now appearing as the *observed* object rather
than as a neighbour.

### Cost of the fix that did work

The floor is not free. Broadening the proposal improves coverage and worsens
the weight tail: Pareto k median 0.469 → 0.564, the fraction above 0.7 goes
9.10% → 32.13% and above 1 goes 2.42% → 11.70% (both arms finite on
essentially every row, so this is like for like). More atoms are reachable,
and some of them arrive with large weights. On the evidence here that trade
is worth taking — it removed the collapse outright and cost no accuracy on
the converged rows — but it is a real cost and it argues against pushing the
percentile higher without measuring this.

### Files and outputs

No code changed. Outputs under
`$DATA_DIR/.../ten_k_floor50_16628938/` and the comparison scripts in the job
scratch. The old arm is `production_lru_v1/part_00` rows 0-9999, reused
rather than re-run.

### Validation

- Job 16628938 COMPLETED, exit 0, 39m03s, 2229.55 s of inference for 10,000
  rows.
- The new arm's `result.json` records `proposal_source =
  proposal_floor50_v1`, its own `proposal_cache_sha256`, and
  `runtime_cache_compatibility = audited_two_entry_response_lru_v1+run_stage:
  ...`, so the run states which proposal and which code it used.
- Reusing production as the old arm is sound but not bit-exact: the current
  tree reproduces production on row 142230 to 5e-5 relative, with
  `draw_counts` and `unique_counts` identical, the difference being
  compiled/chunked floating-point reduction order. Irrelevant at the scale of
  the effects here, and the 10k arm used production's own settings.

### Limitations

- No shear estimate exists from either arm, so nothing here constrains `m`.
- One injected shear, one draw seed per arm, 10,000 of 500,000 rows.
- The claim that the extreme rows are converged rests on their stability
  under a five- to eight-fold change in the atom set, which is strong but is
  not a proof that both arms are not wrong in the same way.

### Next steps

- The obstruction is two objects, and they are bright, large and very
  elliptical. Establish whether the likelihood or the response emulator is
  out of domain there, the way the bright neighbours were in the 2026-07-05
  work. That is a likelihood/emulator question, not a sampler one, and it is
  where the next effort belongs.
- Do not pursue a better sampler for this. The rows that matter are already
  converged.
- Decide the policy question a principled cut implies: if these objects are
  outside the model's domain they must be rejected by a stated rule and
  reported, not dropped because they are inconvenient.
- The weight tail (Pareto k > 1 on 11.7% of rows) is now the sampler's
  leading defect, and removing the exact stratum from the mixture before
  drawing remains the candidate v1.4 for it.
## 2026-09-21 — the proposal cache is rebuilt at the median, and the disk driver stops refusing its own cache

Owner asked to rebuild the proposal cache and run inference on one row under
the new default. Doing so surfaced a blocker created by the entry below: the
V3.6 disk driver refused the 24m-atom cache outright, so no inference could
run at all.

### The cache-reuse gate was refusing everything

`scripts/run_disk_inference.py` records a sha256 of every `sbsi/*.py` and of
itself in the preparation identity, and `check_prepared_identity` allows
exactly one pre-audited pair of files to differ. The v1.3-infer commit
(41ff566) changed `sbsi/catalogue_sampling.py` and `sbsi/catalogue_null.py`,
which made the changed set four files instead of two:

```
GATE REFUSED -> unapproved preparation/runtime implementation difference
```

Neither file can affect a cached artifact. `sbsi/disk_inference_store.py` does
not import `catalogue_null` at all — it holds `run_adaptive_section5`, which
runs long after the cache is written — and `assemble` reaches
`catalogue_sampling` only for `ProposalCoordinateTable`'s constructor and
`save`, whose diff in 41ff566 is a single `_fractional_mask` →
`fractional_mask` rename. The refusal was a false alarm from a proxy check,
not a real identity difference.

`check_prepared_identity` now carries `RUN_STAGE_IMPLEMENTATION`, the set of
files neither preparation nor assembly executes, and permits the changed set
to include them on top of the audited pair. Their content is deliberately
**not** pinned: the inference releases change these two files by design, and
pinning them would tie a cache to whichever release happened to build it.
Everything else is unchanged — a difference in `disk_inference_store.py`, in
any scientific input, or in the four hash-pinned preparation functions is
still refused. The compatibility string returned now names the run-stage files
that differ, and lands in the run's `result.json` as
`runtime_cache_compatibility`, so a run says which code differed from its
cache rather than merely asserting it was allowed.

This relaxes a proxy for a check, not the check itself: `run` separately
verifies every cached artifact against its recorded sha256, and that is
untouched.

### The proposal cache is rebuilt, not re-floored

The assembled proposal table is a *summary* — the flow draws are already
reduced to a per-atom mean and standard deviation, and the floor is applied
afterwards. Changing the floor therefore needs neither the flow nor a GPU.
It cannot be done by re-running `assemble`, which demands byte-identical
implementation across all 36 files and whose own source is hash-pinned, so
`scripts/rebuild_proposal_cache.py` rebuilds only the proposal table, from the
same shard arrays in the same order, into a new directory. Every other
assembled artifact keeps its recorded hash.

The script calls `coordinate_table` — the function `assemble` itself called —
to recompute the shipped table, and **refuses to write unless that
reproduction is exact**. It was:

```
VERIFIED rebuild reproduces .../disk_assembled_v1/proposal exactly
REBUILT atoms=24000000 output=.../proposal_floor50_v1
  measured_ngmix_g1:            floor binds on 50.0% of atoms
  measured_ngmix_g2:            floor binds on 50.0% of atoms
  measured_flux_radius:         floor binds on 50.0% of atoms
  measured_flux_from_mag_auto:  floor binds on 50.0% of atoms
```

Binding on exactly half the atoms is what a median floor must do by
construction, in all four coordinates; it is a consistency check on the
rebuild, not a result.

This is a stronger statement than the 2026-09-21 measurement below could
make. That one re-floored the finished table in memory, which cannot
reproduce a fresh build for a *fractional* coordinate: the floor ranks atoms
by `sigma / |x|`, an order the earlier absolute floor can permute, so the
agreement held only up to the atoms that floor already bound. This rebuild
starts from the unfloored shard summaries, so the flux coordinate is floored
correctly for the first time.

### Running one row against it

`run_disk_inference.py run` gains `--proposal`, a table to draw from instead
of the prepared one. The prepared table is still loaded and hash-verified
either way; the override only changes which atoms the proposal puts in reach,
never what any atom is worth. The run records `proposal_source`,
`proposal_cache_sha256`, `proposal_metadata` and the prepared table's own hash
separately, so a result cannot be mistaken for one drawn from the cache as
built.

`NUMERICS` still points at `configs/inference_v1_2_16k.json` and must: it is
hashed into the preparation identity, so repointing it would invalidate the
cache irrecoverably. A run with `--proposal` is therefore correctly labelled
`pipeline_release="custom"` over base `v1.2-infer-16k` — the v1.2-16k
estimator drawing from v1.3-infer's proposal. The floor is recorded in the
proposal metadata rather than the release string.

### The one-row result: the proposal moves the answer by about a factor of two

Row 142230 — the worst-curvature row of the three in the atom-map study, not
a typical one — run twice under the same estimator, the same draw seed 8701
and the same 16,384-draw ladder, differing only in which proposal table the
draws come from (jobs 16628887_0 and 16628887_1).

| final rung, K=16,384 | prepared, 1st pct | rebuilt, median | |
|---|---|---|---|
| score `g1` | 2043.23 | 1037.49 | **−49.2%** |
| score `g2` | −1210.36 | −552.76 | **+54.3%** |
| information diag `g1` | −2.708e6 | −1.842e6 | −32.0% |
| information diag `g2` | −2.134e6 | −1.003e6 | −53.0% |
| distinct atoms drawn | 2047 (12.5%) | 7897 (48.2%) | **3.9x** |
| effective sample size | 3.93 | 7.97 | 2.0x |
| largest single weight | 0.435 | 0.243 | 1.8x better |
| Pareto k | 1.79 | 2.72 | worse |

So the answer to the question the entry below left open — whether the floored
proposal moves the Hessian — is **yes, decisively**. The score halves and the
curvature falls by a third to a half. The proposal is not a free choice.

Three of the four weight diagnostics improve, and the count that motivated the
floor improves most: the old proposal spent 16,384 draws on 2,047 distinct
atoms, drawing the same ones over and over, while the rebuilt one reaches
nearly four times as many.

The old proposal's ladder also shows the failure directly:

```
   K      score g1      ESS   max wt frac   rel err   pareto k
  512     2178.3775     4.48        0.4130     0.0007      14.99
 1024     2179.0662     6.01        0.3459     0.0004       4.28
 2048     2179.0939    10.85        0.1780     0.0003       2.36
 4096     1960.7926     1.02        0.9918     0.1923       2.11
 8192     2015.3075     1.88        0.7032     0.1057       1.89
16384     2043.2304     3.93        0.4352     0.0608       1.79
```

The first three rungs agree to four significant figures and report a relative
error of 3-7 parts in ten thousand — and then the answer moves by 10% when a
new atom finally arrives at K=4096, and one draw carries 99.2% of the weight.
That apparent precision was the proposal re-drawing the same narrow set, not
convergence. The rebuilt proposal shows no such false plateau.

**Neither run is a converged estimate, and this does not validate v1.3.**
Pareto k is 1.79 and 2.72; above 1 the importance weights have no finite mean,
so both numbers are untrustworthy in the strict sense and the ESS figures of
4-8 out of 16,384 draws are themselves unreliable. What the comparison
establishes is a lower bound on how much the proposal matters, not which
answer is right.

**Uncertainties.** There are no seed-to-seed error bars here: one row, one
draw seed per arm. The only stability handle within a run is the spread of
the top three ladder rungs (K=4096, 8192, 16384), which is 4.1% and 3.9% of
the mean for the prepared arm and 4.6% and 6.6% for the rebuilt arm, on `g1`
and `g2` respectively. The gap between the two proposals is roughly ten times
that spread, so the difference is not draw noise. Proper error bars need the
run repeated over draw seeds, which has not been done.

Outputs, including the comparison script and its output, are under
`$DATA_DIR/.../one_row_16628887/`.

### Files

- `scripts/rebuild_proposal_cache.py` — new.
- `scripts/run_disk_inference.py` — `RUN_STAGE_IMPLEMENTATION` and the
  widened gate; `--proposal`; the proposal provenance in `result.json`; the
  atom-count check on an overridden table. The four hash-pinned preparation
  functions are untouched and verified so.
- `tests/test_disk_prepared_identity.py` — new; nothing covered this gate
  before.

### Validation

- `pytest tests -q` (job 16628892, same six pre-existing uncollectable files
  excluded as in the entry below) → **285 passed**, up from 278.
- `pytest tests/test_disk_prepared_identity.py -q` → **7 passed**. They cover
  the unchanged tree, a changed scientific input, a preparation-stage change
  (refused), run-stage-only changes (accepted and named), the audited pair
  alone keeping its original string, a run-stage change without the audited
  pair (refused), and the preparation-function pin itself.
- Rebuild job 16628880 → COMPLETED, 1m51s, 4.8 GB peak RSS, CPU only.
  Exact reproduction of the shipped table verified before writing.
- One-row inference array 16628887 → both tasks COMPLETED, one GPU at a time
  (`--array=0-1%1`), 14-17 s of inference each after a 50-67 s cache load.
  Both `result.json` files record
  `runtime_cache_compatibility = audited_two_entry_response_lru_v1+run_stage:
  sbsi/catalogue_null.py,sbsi/catalogue_sampling.py`, distinct
  `proposal_cache_sha256` values and the floor in `proposal_metadata`, so each
  run states which proposal it drew from.

### Limitations

- The gate is a file-hash proxy and remains one. `RUN_STAGE_IMPLEMENTATION` is
  a hand-maintained list: if a future change makes `assemble` depend on
  `catalogue_null`, or on more of `catalogue_sampling` than the table
  constructor, the list must be revisited. The import surface is checked in
  this entry, not enforced by a test.
- The rebuilt table is verified to reproduce the shipped one at the *build*
  floor. That validates the path, not the median floor itself, whose
  justification remains the three-row measurement in the entry below.
- Nothing rebuilds the prepared shards; the flow summaries are reused as
  produced on 2026-09-20.
- One row, one draw seed per arm, and the hardest of the three rows studied.
  No seed-to-seed error bars, and nothing here says the rebuilt arm's number
  is closer to the truth — only that it is drawn from a far wider set of
  atoms and shows none of the false convergence the prepared arm does.
- Both arms fail the Pareto-k criterion by a wide margin. Whatever fixes that
  is a larger change than a proposal floor.

### Next steps

- Repeat the one row over several draw seeds to get real error bars on the
  ~50% shift, and add a second and third row so the shift is not read off the
  worst-curvature row alone.
- Pareto k above 1 in both arms is the binding problem now, not the floor.
  Removing the exact stratum from the mixture before drawing (noted below as
  a possible v1.4) recovers wasted draws but does not by itself fix a weight
  distribution with no finite mean.
- If the disk path is to run under a release string rather than `custom`, the
  preparation identity has to stop hashing `NUMERICS`, which means rebuilding
  the prepared shards. That is a deliberate, GPU-priced decision and is not
  taken here.
## 2026-09-21 — v1.3-infer: the floored proposal becomes the default, and the sampler paths nothing runs are deleted

Owner asked to clean the repository up, delete the variants and dead code, and
make the floored sampler the default under a new inference identity. The
measurement that justifies the floor is the entry below; nothing here re-derives
it and no new science run was made.

### Is it 1.3?

Yes. `v1.1-infer` named the defensive-mixture estimator and was the default.
`v1.2-infer` named the tilted-stratified estimator and was never made the
default. This change is neither of those: it keeps `v1.2-infer`'s estimator
exactly and replaces the proposal that estimator draws from. A different
proposal is a different numerical result, so it takes its own release string
rather than quietly redefining `v1.2-infer`, and since it is now what an
ordinary run gets, it is the default. `v1.3-infer`.

The likelihood is untouched: `v3.2-like`, as for both predecessors.

### Behavior change

`configs/inference.json` is the file every run loads unless told otherwise, so
that file now holds `v1.3-infer`:

| | v1.1-infer | v1.2-infer | **v1.3-infer** |
|---|---|---|---|
| estimator mode | mixture | tilted_stratified | **tilted_stratified** |
| exact stratum | — | 1,024 | **1,024** |
| draw ladder top | 16,384 | 8,192 | **16,384** |
| proposal scatter floor | 1st percentile | 1st percentile | **50th percentile** |
| fractional floor | none | none | **measured_flux_from_mag_auto** |

The previous default is preserved verbatim as `configs/inference_v1_1.json`;
`configs/inference_v1_2.json` and `configs/inference_v1_2_16k.json` are
untouched. No release configuration was deleted: completed runs name them, and
deleting them would destroy the provenance of results already recorded here.

**This changes what a default run does.** It is the first commit in this series
that does; the entry below deliberately changed no default.

### Code

`sbsi/catalogue_sampling.py`

- `_fractional_mask` is now public `fractional_mask`: two modules build the
  mask, and a private name crossing that boundary was the wrong signal.
- Deleted `draw_global`, `uncertainty_candidates`, `_build_uncertainty_mips_tree`
  and the MIPS tree state on `DefensiveLocalProposal`, `draw_priority`,
  `select_priority_batch` and `_priority_row_seed` — 329 lines. Only tests
  reached any of them.
- **`from_flow`'s default `dispersion_floor_percentile` deliberately stays at
  1.0.** The floor is a property of a catalogue, not of the library: 50 was
  measured on this prior and on three rows of it. Making it a library constant
  would generalise that measurement further than it goes, and would silently
  change `run_catalogue_closure.py` and `build_model_cache_from_zero_view.py`,
  which were never measured. The value that decides a run is the one its
  release configuration names.

`sbsi/catalogue_null.py`

- `priority_stratified` removed from `STRATIFIED_MODES`, with its tilt check,
  its single-rung ladder refusal and its draw branch. It is sampling without
  replacement; no configuration, job script or run ever selected it, and its
  own validation refused the draw ladder every configuration uses. The
  three-way branch is now two: the tilted complement or the flat one.

`scripts/run_inference.py`

- `--proposal-dispersion-floor-percentile` and `--proposal-fractional-floor`,
  read from the `proposal` block of the release configuration, passed to
  `from_flow`, and echoed into the resolved document only when they differ from
  the historical setting — so a v1.1 or v1.2 run still resolves to its own
  release name rather than `custom`.
- The floor is recorded in the **proposal cache identity**. A cache built at
  one percentile is a different proposal from the same table built at another,
  so a v1.3 run refuses a cache floored at 1% instead of reusing it silently.
  A cache written before this change carries no floor key and is read as the
  historical 1.0, which is what it was built with.
- `priority_stratified` removed from the mode choices and its help text.

`scripts/merge_sharded_prior_model_qmc.py`

- `_global_dispersion_floor` deleted. It was a second copy of the floor that
  predated the shared helper and hardcoded the first percentile, and it is the
  one that actually built the 24m-atom production cache. It now calls
  `floored_dispersion` with `--dispersion-floor-percentile` (default 50.0) and
  `--fractional-floor` (default `measured_flux_from_mag_auto`). The fallback
  it passes is identical to the one it used before.

`jobs/job_inference.sh` header names v1.3-infer.

### What flipping the default broke, and what was done about it

Eight job scripts pass `configs/inference.json`, so every one of them changed
meaning at once. Three needed anchoring rather than following:

- `jobs/job_inference_stratified_screen.sh` compares a mixture arm against a
  stratified one and asserts the mixture arm "overrides nothing and must come
  back labelled `v1.1-infer`". Against a default that now declares a mode of
  its own, that arm would have come back `custom` and the screen would have
  compared two custom arms. It now names `configs/inference_v1_1.json`.
- `jobs/job_inference_candidate_ladder.sh` sweeps K around the release point
  and rests on "the K=16,384 arm is the base configuration, so it is checkable
  rather than merely plausible". The base configuration is now K=1,024, which
  moves that anchor off the sweep entirely. It now names
  `configs/inference_v1_1.json`.
- `priority_stratified` was still a selectable MODE in the screen script and
  would have failed at run time with `unknown estimator mode`. Removed.

The rest follow the default deliberately. The three `job_prepare_*_mock*.sh`
jobs build caches from the configuration, so they now build a **floored**
proposal cache — which is the rebuild v1.3-infer needs, already wired. The
`job_complement_*` and `job_likelihood_landscape.sh` diagnostics override
`--proposal-candidates` explicitly, so the K change does not reach them.

Stale wording corrected in `scripts/run_inference.py`: the module docstring no
longer claims to run v1.1-infer, the resolved-document comment no longer says a
matching run reports `v1.1-infer` specifically, and the initial-center proposal
check no longer attributes that requirement to v1.1 alone.

### Validation

- `pytest tests -q` (job 16628677) → **278 passed**, with six test files
  excluded by `--ignore`. Those six do not collect *at HEAD as well*:
  `test_api.py`, `test_cli.py`, `test_flow_coupling_target.py`,
  `test_image_closure.py`, `test_learned_retrieval_diagnostic.py` and
  `test_measurement_model.py` all fail on `ImportError: cannot import name
  'ConditionalMeanFlowRA' from sbsi.measurement_model`. The symbol is absent
  from that module at HEAD and this commit does not touch it; the pickaxe log
  puts its removal in the bulk import `a010491`, which rewrote
  `sbsi/measurement_model.py` (-422 lines) without updating the tests that
  import it. Pre-existing and **not repaired here** — restoring the
  realisation-aware head or retiring those tests is a measurement-model
  decision, not a sampler one. A collection error aborts the whole run
  (job 16628669), which is why they are excluded rather than left to fail.
- `pytest tests/test_catalogue_sampling.py tests/test_release_config.py
  tests/test_proposal_atom_map.py tests/test_inference_provenance.py -q` →
  105 passed.
- All four release configurations round-trip to their own name, checked
  directly through the runner's own `_resolved_pipeline_config`:
  `inference.json` → v1.3-infer (floor 50.0, fractional on
  `measured_flux_from_mag_auto`), `inference_v1_1.json` → v1.1-infer (floor
  1.0, none), `inference_v1_2.json` → v1.2-infer, `inference_v1_2_16k.json` →
  v1.2-infer-16k. None resolves to `custom`.
- Two new tests in `tests/test_release_config.py`:
  `test_v1_3_is_the_default_and_carries_the_floored_proposal` and
  `test_the_floor_reaches_the_proposal_cache_identity`. The v1.1 test now
  guards `configs/inference_v1_1.json` and additionally asserts that no floor
  key leaks into its resolved document.
- `tests/test_priority_sampling.py` deleted (226 lines); the `draw_global` and
  direct-MIPS tests deleted from `tests/test_catalogue_sampling.py`.
- Figures regenerated at the committed code under the floored proposal, job
  16628553: rows 142230 / 409188 / 3563 reach 88.8% / 91.1% / 92.0% of the
  captured mass with 38% / 18% / 17% of draws wasted. Reports are byte-identical
  to job 16628512, so the path reproduces exactly. Copies in `SBSI/plots/`.

### Limitations

- **No inference has been run under v1.3-infer.** The identity is defined and
  defaulted on the strength of the proposal measurement in the entry below —
  three rows, eight draw seeds, and the mass the estimator *reaches*. What the
  estimator then *reports* with that mass is unmeasured.
- The existing prepared caches under the V3.6 run directory were built at the
  1st percentile. Under v1.3 they are now correctly refused rather than reused,
  which means a v1.3 run needs its proposal cache rebuilt or re-floored. That
  rebuild has not been done or costed here.
- The 50th percentile is the better of three settings tried, not an optimised
  value, and the three rows were chosen as the worst-curvature rows rather than
  sampled.
- `fractional_floor_targets` names a coordinate. A model whose targets do not
  include `measured_flux_from_mag_auto` will be refused by `fractional_mask`
  rather than silently left unfloored. That is the intended failure, but it
  couples the default configuration to the current target set.
- Deleting `priority_stratified` removes the only without-replacement option.
  Nothing used it, but recovering it means reverting this commit, not flipping
  a flag.
- Prior resolution and the finite-difference/autodiff swap remain untouched;
  both are larger than the sampler defect this series has been fixing.

### Next steps

- Rebuild or re-floor a proposal cache at the median and run the inference on
  one row under v1.3-infer. This is the first identity whose proposal reaches
  most of the mass, and it is still unknown whether that moves the Hessian.
- Removing the exact stratum from the mixture before drawing would recover the
  remaining ~24% of wasted draws directly. It changes the estimator, so it
  would be v1.4, not a proposal change.

## 2026-09-21 — Flooring the proposal's predicted scatter: the ranking defect and the wasted draws are one fix

Owner asked to fix the evident problems from the review before the larger
questions. Three were in scope: the proposal dispersion floor, the atom map
reproducing a sampler production does not run, and retiring the score and
sampler variants the floor makes redundant. Prior resolution and the
finite-difference/autodiff swap are explicitly **not** in this entry; each
needs its own run.

### The measurement

The floor is the existing `dispersion_floor_percentile` on
`ProposalCoordinateTable`, production value `1.0`. Re-floored tables were
compared against the shipped one under the *configured* draw
(`tilted_stratified`, with replacement, exact stratum left in the mixture),
16,384 draws, rows 142230 / 409188 / 3563, jobs 16628412 and 16628488:

| proposal scatter | exact-stratum share of `q` | usable draws | posterior mass reached |
|---|---:|---:|---:|
| shipped — 1st pct, absolute | 0.88 / 0.90 / 0.89 | 11.1% | 57.0%, sd 31.9 |
| **50th pct, fractional flux** | **0.39 / 0.18 / 0.16** | **75.7%** | **92.1%, sd 2.9** |
| 90th pct, fractional flux | 0.031 / 0.015 / 0.011 | 98.1% | 0.4% |

Per row, over 8 independent draw seeds (`+/-` is the standard error on the
mean; the proposal is deterministic, so only the draw varies, and the
exact-stratum share has no seed dependence at all):

| row | reached, shipped | reached, floored | usable draws, shipped | usable draws, floored |
|---|---:|---:|---:|---:|
| 142230 | 16.1 +/- 0.2% | **89.2 +/- 0.3%** | 12.4 +/- 0.1% | 61.2 +/- 0.1% |
| 409188 | 62.9 +/- 0.0% | **93.4 +/- 0.9%** | 10.3 +/- 0.1% | 82.3 +/- 0.1% |
| 3563 | 92.0 +/- 0.0% | **93.7 +/- 0.8%** | 10.7 +/- 0.1% | 83.7 +/- 0.1% |

Seed noise is below 1% everywhere, so the gap is many times its own
uncertainty. The second effect matters as much as the mean: the row-to-row
spread collapses from sd 31.9 to sd 2.9. The shipped proposal is erratic —
adequate on 3563, broken on 142230 — while the floored one is uniformly good.
The worst importance weight on a drawn mass atom is 4.7x / 1.5x / 1.7x,
against a shipped run that drew *no* mass atom outside the exact stratum at all.

Two defects with one cause and one fix:

1. **The ranking defect.** The proposal divides each residual by the atom's own
   predicted scatter, so a vague atom is judged on a loose tolerance and a
   sharp one on a tight tolerance. Faint atoms are vague and outnumber bright
   ones ~340:1, so coincidence wins: the shipped top-8192 for row 409188 has a
   median true `r` of 27.13 against an observation at 18.33, with 91.7% of
   those atoms fainter than `r = 26`. Flooring the scatter at the median puts
   the median true `r` at 18.76 with 0% beyond `r = 26`.
2. **The wasted budget.** The exactly summed stratum is *not* removed from the
   mixture the sampler draws from, so a draw landing inside it is spent and
   contributes nothing (`catalogue_null._exact_plus_complement`). With ~88% of
   `q` inside that stratum, roughly 7 of every 8 draws were thrown away.
   Flattening the proposal fixes this as a side effect, not by a separate
   change: waste falls from ~89% to ~24%.

**The faint cluster is not uniformly eliminated.** In the top-8192 by `q`, the
fraction of atoms fainter than `r = 26` goes 91.7% -> 0% on row 409188 and
91.8% -> 0% on row 3563, but only 95.7% -> 48.0% on row 142230, and the
regenerated Panel 2b for that row still shows a visible faint blob. Row 142230
is the brightest observation of the three (measured mag 17.30) and the one
whose exact stratum still holds 0.39 of `q` after flooring. So the floor
removes the faint mode on two of three rows and halves it on the third; it does
not close that question. It does not need to for the estimator's sake there —
89.2% of the mass is still reached — the residual cluster costs draws, not
accuracy.

The 90th percentile is a genuine failure in the other direction — the proposal
becomes so broad it stops finding the mass at all (0.4% reached). The median is
therefore bracketed by two failures rather than picked from a list.

Flux needs a **fractional** floor specifically. Its scatter is floored in
absolute units across five decades of flux, so the floor is set by the faintest
atoms and cannot bind on a bright one: atoms brighter than `r = 20` carry a
median absolute flux scatter 92x above the 1st-percentile floor and 12x above
the median, while in fractional terms their scatter is 0.011 dex against 0.53
dex for atoms fainter than `r = 26`. Ranking that coordinate on `sigma / |x|`
makes the floor mean the same thing at every brightness.

### Code

`sbsi/catalogue_sampling.py`

- New `floored_dispersion(values, dispersion, *, percentile, fractional,
  fallback)` — the floor, lifted out of `from_flow` and given a fractional
  mode. The percentile is taken on the active atoms' own scatter, so it
  introduces no hand-chosen scale; a coordinate flagged `fractional` is ranked
  on `sigma / |x|` and the floor carried back to absolute units per atom.
- New `_fractional_mask(names, fractional_targets)` — rejects a target name
  absent from the table rather than silently ignoring it.
- `ProposalCoordinateTable.from_flow` gained `fractional_floor_targets` and now
  delegates to `floored_dispersion`. **`dispersion_floor_percentile` keeps its
  default of 1.0**: no production default changes in this commit.
- New `ProposalCoordinateTable.with_dispersion_floor(*, percentile,
  fractional_targets)` — re-floors a cached table with no flow evaluation, and
  records the re-floor in `metadata["dispersion_refloor"]` so a re-floored
  cache cannot be mistaken for a freshly summarised one. For an *additive*
  coordinate this reproduces a fresh build exactly when the new percentile is
  at least the applied one (tested to `atol=0`); for a *fractional* one the
  earlier absolute floor can permute the `sigma/|x|` order, so it agrees only
  up to the atoms that floor already bound — 1% of them at the shipped setting.

`scripts/plot_proposal_atom_map.py`

- New `production_draw` replaces the priority race: one fixed-width uniform
  record per draw from a generator seeded by `(seed, object_id)`, second column,
  inverse-CDF on the cumulative mixture — sampling **with replacement**, with
  the exact stratum left in place and draws landing there counted as wasted.
  This is what `estimator_mode = tilted_stratified` actually reaches. The
  previous `select_priority_batch` reproduction belongs to `priority_stratified`,
  which nothing runs; the correction is in the entry below.
- Reports coverage `1 - (1 - q)^M` where the priority version reported an
  inclusion probability. With replacement there is no inclusion probability.
- `--sampler` and `--score` are gone, and with them `priority_race`,
  `stratified_race`, `slot_accounting`, `common_metric_scale`,
  `common_metric_score`, `mixture_from_score` and the `FIFTY_FIFTY_*` constants.
  The fifty-fifty split and the shared-metric score were both approaching the
  floor from outside; neither existed in `sbsi/`. Replaced by
  `--floor-percentile` and a repeatable `--fractional-floor TARGET`.
- The report now carries a `dispersion_floor` block and a `sampler` block
  naming the mode, the wasted fraction and the worst weight on a drawn mass
  atom.
- **Bug fixed, caught by the first regenerated figure:** `n_drawn` holds the
  number of *distinct* atoms the draw reached (7,604 on row 142230), and six
  sites used it as if it were the draw budget (16,384). The figure therefore
  claimed "82% of draws wasted" where the true share is 38%, and every
  `1 / (M q)` weight in the report was inflated by 2.2x. Renamed to `n_unique`
  and the budget sites switched to `PRODUCTION_DRAWS`. The independently
  written measurement job was never affected, which is why the two disagreed
  and the error surfaced; job 16628412's numbers stand.

`scripts/run_proposal_atom_map.sh` takes `[FLOOR_PERCENTILE]
[FRACTIONAL_TARGET...]` instead of the two retired mode arguments.

### Validation

- `pytest tests/test_catalogue_sampling.py tests/test_proposal_atom_map.py -q`
  → **77 passed**. 9 new floor tests, 7 new `production_draw` tests; 29 tests
  covering the retired helpers removed. Net −124 lines across the five files.
- The pivotal test is
  `test_fractional_floor_binds_on_bright_atoms_where_an_absolute_floor_cannot`:
  on a toy catalogue spanning five decades, an absolute floor at the median
  leaves the bright atoms' flux tolerance *bit-for-bit unchanged*, while the
  fractional floor lifts it by a measured factor of 4.59 and puts those atoms
  exactly on the median ratio. Every threshold in the new tests was measured
  first and the measured value recorded in a comment; none was guessed.
- Job 16628412 — the floor comparison; job 16628488 — the 8-seed error bars.
- Jobs 16628480 / 16628481, regenerated after the label fix as 16628513 /
  16628512 — the repaired atom map end to end on all three rows.
  **Cross-check:** 16628480 reports `REACHED 11/32 captured=0.9350
  reached=0.1493` for row 142230, i.e. 16.0% of the captured mass, and
  16628481 reports `reached=0.8303`, i.e. 88.8%. Both match job 16628412's
  independently written measurement of the same row and settings.

### Limitations

- **Nothing in production changed.** `dispersion_floor_percentile` still
  defaults to 1.0 and no config was edited. Adopting the median floor means
  rebuilding or re-flooring the proposal table for a run and is a separate,
  owner-authorised step.
- Three rows, chosen as the worst-curvature rows, not a random sample. They
  are not evidence about the other ~500k. The pooled sd above is a spread over
  three rows, so it is an indication, not a population estimate.
- The 50th percentile is the better of three settings tried, not an optimised
  value. It was not tuned against any target number, and no intermediate
  percentile was scanned.
- This addresses the finite-draw integration error only — the smallest of the
  three problems ordered in the entry below. Exact 24m summation on row 142230
  is already non-concave, so no amount of sampler repair alone produces a
  positive information matrix. Reaching 92.1% of the captured mass instead of
  57.0% does not change that conclusion.
- The floor is a proposal-only heuristic, as `doc/V36_INFERENCE_REVIEW.md`
  sanctions: it changes which atoms are proposed, never what they are worth.
  The likelihood and the target are untouched.
- An overflow in the first version of the fractional ratio (a shear component
  passes through zero, where `sigma / |x|` is not finite) was caught by job
  16628412's own warning output and is fixed with a regression test; atoms that
  cannot form the quotient keep the absolute floor.

### Next steps

- Run the inference on one row with the median floor to see what the estimator
  does with 92% of the mass instead of 57% — the first number this series has
  produced that could plausibly move the Hessian.
- Prior resolution remains the largest of the three problems and is untouched.
- Swap the h=0.001 finite differences for autodiff (957.16 / 1520.06 against
  390.91 / 766.12 — the derivative error is larger than the quantity).
- Removing the exact stratum from the mixture before drawing would recover the
  remaining ~24% of wasted draws directly, rather than as a side effect. Not
  done here: it changes the estimator, not the proposal.
- Row 142230's residual faint cluster is unexplained; it is the brightest of
  the three observations and retains the largest exact-stratum share.

## 2026-09-21 — Sampler and inference review: two corrections to this log, and the floor that already exists

Owner asked for a review of the whole sampler and inference path with an
explicit instruction to keep any remedy minimal. Two entries below need
correcting, and the remedy this series has been converging on turns out to be
an existing configuration value rather than a new score.

**Correction 1: the atom-map diagnostics model a sampler production does not
run.** The failed production run's own `result.json` pins
`estimator.estimator_mode = tilted_stratified`, which dispatches to
`DefensiveLocalProposal.draw_tilted` and thence to
`WholeCatalogueProxy.draw_uniforms_batch` — inverse-CDF lookup on the
cumulative mixture, i.e. sampling **with replacement**.
`scripts/plot_proposal_atom_map.py` reproduces
`WholeCatalogueProxy.select_priority_batch`, which is priority sampling
**without replacement** and belongs to the unused `priority_stratified` mode.
The script's docstring calls this "the production priority race".

The proposal `q` is identical in both, so every statement in this log about
*where the proposal puts its mass* — the blind ranking, the exact stratum
holding 88% of `q`, the faint-atom preference, the rank bands — is unaffected.
What does not carry over is anything derived from inclusion probabilities:
the Horvitz-Thompson weights (2,930x on row 142230's heaviest missed atom;
2x/2x/6x under the fixed-count draw), the saturation counts, and the
"priority sampling caps every atom at one slot" mechanism in the entry of that
name. With replacement there is no cap and no `tau`; a heavy atom is simply
redrawn many times, which is why the production final-rung median ESS is
893.97 out of 16,384 draws. The `--sampler fifty-fifty` variant added in this
series exists in no production path at all.

**Correction 2: the shared-metric score is a re-derivation of an existing
knob.** `ProposalCoordinateTable.from_flow` already floors each coordinate's
dispersion at `dispersion_floor_percentile`, default and production value
`1.0`, and `doc/V36_INFERENCE_REVIEW.md` already sanctions that floor as a
proposal-only heuristic that does not modify the target. Measured over the
production table (24,000,000 atoms):

| coordinate | 1st pct (the floor) | median | ratio | atoms pinned at the floor |
|---|---:|---:|---:|---:|
| g1 | 0.02916 | 0.34155 | 11.7 | 240,001 (1.000%) |
| g2 | 0.02925 | 0.32227 | 11.0 | 240,001 (1.000%) |
| flux_radius | 0.14543 | 1.2391 | 8.5 | 240,490 (1.002%) |
| flux | 5.2978 | 39.198 | 7.4 | 241,915 (1.008%) |

By true magnitude, the atoms brighter than `r = 20` sit *exactly* at the shape
floor (median g1 dispersion 0.02916, a factor 1.0 above it) and near the size
floor (0.2172, factor 1.5), while atoms fainter than `r = 26` sit a factor
11-12 above it. So the proposal judges a bright atom's shape on a tolerance
twelve times tighter than a faint atom's, and the `-sum log sigma` term pays
the faint atom for the privilege. Flooring at the median instead of the 1st
percentile gives `sigma_eff = max(sigma_j, median)`, which is the
quadrature-floor score of the entry above to within a factor of root two, and
it keeps the normalisation, so it does not inherit the shared metric's defect
of trusting a faint atom's noise-driven central value.

**Flux is the exception and needs the one real code change.** The flux floor
is in absolute units, and flux spans five decades across this prior, so it
cannot bind where it matters: atoms brighter than `r = 20` carry a median
absolute flux dispersion of 484.9, a factor 92 above the floor of 5.30, and
raising the floor to the median (39.2) still leaves them untouched. In
fractional terms the tolerance runs from 0.0106 dex at `r < 20` to 0.5259 dex
at `r > 26`, a factor of 50. Giving a bright atom a 0.3 dex tolerance requires
`sigma` of order 17,000, which no percentile of the absolute distribution
reaches. The flux coordinate therefore needs a *fractional* floor, or
equivalently carrying log-flux as the proposal coordinate.

**The ordering that matters.** Three problems are separately established in
this log and only one of them is the sampler's:

1. *Finite-prior resolution / genuine non-concavity.* Exact summation over all
   24 million atoms on row 142230 gives information
   `[[-1686130.55, 500735.41], [500735.41, -879639.81]]`. No proposal, score
   or draw budget can repair a number computed with no sampling at all.
   Effective posterior atom count is 6.29 and 16.58 on the two bright failures
   against 11,636.98 and 292,088.13 on rows 0 and 1, with a median 88.88%
   single-atom mass, and the dominant atom's identity changes across the
   stencil (8.57% at centre to 97.44% at the +g1/-g2 corner). Measured here,
   only about 0.2% of the prior — roughly 48,000 of 24,000,000 atoms — is
   brighter than `r = 20`, while the failing observations are at `r = 17.3`.
2. *Finite-difference derivative error.* At `h = 0.001` the g1 scores are
   957.16 and 1520.06 against autodiff 390.91 and 766.12.
3. *Finite-draw integration error.* Production information -2,707,895 against
   exact -1,686,131 on the same row.

Everything this series has worked on since the atom map is problem 3, the
smallest of the three, and it cannot by itself produce positive information.

Validation. Jobs 16628308 (dispersion floor by coordinate and by true
magnitude) and the three diagnostics of the entry above, all COMPLETED on
`cluster`, 8 CPUs, no GPU. Production configuration read from
`production_lru_v1/part_00/result.json`. Call-site audit: `draw_global` and
`uncertainty_candidates` have no callers outside `tests/`; `draw_priority` has
two callers but is not the configured mode. No repository code changed; the
atom-map suite is unchanged at 61 passed.

Limitations. This is a reading and measurement review, not an experiment: the
median-percentile floor has not been run through the estimator, so there is no
reached-mass, ESS or information number for it, and no claim is made that it
produces positive curvature. The re-floor is valid post-hoc on the stored
dispersions only for percentiles at or above the 1.0 already applied. Three
rows, centre node, one seed throughout.

Next steps, in the order the three problems deserve. Establish whether the
bright-object effective atom count can be raised at all — a magnitude
stratified prior subsample with exact compensating masses `1 / (N p_keep)` is
the only lever that is not a model change, and the full store holds about six
times more bright atoms than the uniform 24m subset kept. Replace the
`h = 0.001` finite differences with the autodiff path that
`probe_disk_derivative_reference.py` already implements. Only then revisit the
proposal, and there change `dispersion_floor_percentile` plus a fractional
flux floor rather than adding a score variant; retire `--sampler fifty-fifty`
and the `--score common-metric` branch when that happens, and fix the atom
map's docstring and draw to match `tilted_stratified`.
## 2026-09-21 — The faint cluster in Panel 2b is real: both scores omit a different half of the uncertainty

Owner asked why a large cluster of ranked draws still sits in the faint, small
corner of the truth panel after the shared-metric score. It is not a plotting
artefact and not blending. The entry below overstated the shared metric's
effect on row 142230, and this corrects it.

**What the figure is showing.** Blue crosses are draws from the ranked
stratum; grey crosses are the defensive uniform half, which is a random
catalogue sample and belongs in the faint corner by construction. On row 142230
under the shared metric, the ranked draws split almost evenly: 44.5% at true
`r < 20` holding 0.175 of the tail mass, and 53.6% at true `r > 26` holding
0.215. The faint half therefore carries *more* proposal weight than the good
half. The same split holds over the top 200,000 atoms by `q` (43.7% against
54.2%), so this is a property of the proposal, not of the draw. The previous
entry's claim that the blue cloud now sits on the observation holds for rows
409188 and 3563 (median drawn true `r` = 19.31 and 19.93 against observations
at 18.33 and 18.34) but not for row 142230 (26.44 against 17.28).

**Why faint atoms win.** They are not faint in the coordinates the score sees.
Measured over a 300,000-atom random sample, with the zero point
`flux = 10^(-0.4 (r - 30))` that the observations satisfy to 0.05 dex:

| true `r` | n | predicted − implied flux | sigma_flux | sigma_g1 | sigma_Rflux | Pdet |
|---|---:|---:|---:|---:|---:|---:|
| 0–20 | 602 | −0.02 dex | 0.01 dex | 0.029 | 0.24 px | 0.999 |
| 20–22 | 2,899 | −0.00 | 0.02 | 0.029 | 0.24 | 0.998 |
| 24–26 | 76,018 | +0.12 | 0.16 | 0.297 | 1.06 | 0.991 |
| 26–27 | 85,210 | +0.18 | 0.29 | 0.359 | 1.39 | 0.879 |
| 27+ | 119,788 | +0.43 | 0.93 | 0.359 | 1.39 | 0.013 |

A bright atom's predicted measurement is sharp: 0.01 dex in flux, 0.029 in
ellipticity, 0.24 pix in size. A faint atom's is vague, and at `r > 26` its own
quoted scatter (0.36 in shape, 1.39 pix in size, 0.29–0.93 dex in flux) equals
or exceeds the shared tolerance `(0.342, 0.322, 1.239 pix, 0.337 dex)`. The
shared-metric score discards the per-atom scatter entirely, so it reads each of
those noisy central values as a real match. There are about 48,000 atoms
brighter than `r = 20` against roughly 16 million fainter than `r = 26`, a
340:1 numerical advantage, so even a small per-atom chance of a coincidental
match dominates the total. Detection suppresses only the `r > 27` end
(`Pdet = 0.013`); the winning band is `26 < r < 27`, where `Pdet = 0.879`.
Neighbour light is not the mechanism: the ranked draws sit at the catalogue's
median crowding (`nbr_flux_near` 0.75, i.e. neighbour-flux-to-noise about 4.8)
while the mass-carrying atoms are far more crowded (1.17, 2.34, 2.60, i.e. 14,
217 and 397).

**Both scores omit half the uncertainty, and they omit different halves.**
Production divides the residual by the atom's own predicted scatter alone. That
scatter describes the *prediction*; it does not include the noise on the
*observation*. So a bright atom that quotes 0.01 dex is judged at 8 sigma for a
0.08 dex miss and is destroyed by its own error bars, while a vague atom fits
anything — median score −14.05 against −15.95 on row 142230, a 1.9 nat margin
that the 340:1 count turns into a rout. The shared metric fixed the
normalisation but replaced per-atom scatter with a single tolerance, which is
wrong in the opposite direction: it trusts a faint atom's noise-driven central
value as much as a bright atom's sharp one.

**Adding them in quadrature removes both pathologies.** With
`sigma_eff^2 = sigma_shared^2 + sigma_atom^2` and the normalisation retained,
measured on the same three rows (ranking only; the sampler was not rerun):

| row | score | top-8192 median true `r` | frac `r > 26` | top-8192 median true `R_e` | mass atoms in top-1024 |
|---|---|---:|---:|---:|---:|
| 142230 (obs 17.28, 1.45") | production | 27.07 | 95.7% | 0.09" | 11/32, 16.0% |
| | common-metric | 26.39 | 54.0% | 0.17" | 7/32, 58.7% |
| | quadrature | **18.54** | **0.0%** | **1.57"** | 10/32, 61.1% |
| 409188 (obs 18.33, 1.59") | production | 27.13 | 91.7% | 0.09" | 26/32, 62.9% |
| | common-metric | 18.87 | 14.3% | 1.59" | 20/32, 80.1% |
| | quadrature | **18.80** | **0.0%** | 1.63" | 15/32, 67.2% |
| 3563 (obs 18.34, 1.17") | production | 27.01 | 91.8% | 0.09" | 25/32, 92.0% |
| | common-metric | 19.20 | 28.2% | 1.02" | 17/32, 82.8% |
| | quadrature | **19.06** | **0.0%** | 1.16" | 11/32, 65.6% |

The faint cluster disappears completely on all three rows and the top-8192
median size matches the observation to better than 0.12 arcsec. The exact
stratum's share of captured mass becomes more even across rows (61%, 67%, 66%)
but is lower than the shared metric's best rows, so the reached-mass comparison
needs the sampler rerun before anything is claimed about it.

Validation. Jobs 16627872 (truth by draw class, both scores), 16627902 (rank
bands by true magnitude, tail mass by bin) and 16627932 (predicted flux against
true magnitude, three candidate rankings), all COMPLETED on `cluster`, 8 CPUs,
no GPU. Scripts under the job scratch directory; no repository code changed, so
the test suite is unchanged at 61 passed. An earlier version of the first
diagnostic tested for blending by comparing `nbr_flux_near` against 10% of the
observed flux; `nbr_flux_near` is `log10(1 + near_flux / aperture_rms)`, not a
flux, so that test was meaningless and its "0 of 8192 blended" result is
withdrawn.

Limitations. Three rows, centre node, one seed, ranking only — the quadrature
score has not been run through the sampler, so no reached-mass or estimator
variance number exists for it, and nothing here says it is better where it
matters. The zero point 30.0 is inferred from the observations themselves
rather than read from a header. The shared shape tolerance 0.342 remains as
wide as the whole ellipticity distribution under every variant.

Next steps. Wire the quadrature tolerance in as a third `--score` choice, with
a test pinning that it reduces to production as the shared floor goes to zero
and to the shared metric as the per-atom scatter goes to zero, then rerun the
atom map and the reached-mass table. Only then compare the three on captured
mass and Horvitz-Thompson weights. Making the shape tolerance depend on the
observation's own brightness remains open and untried.
## 2026-09-21 — A shared-metric score fixes the ranking: 17% to 88% of the captured mass on the worst row

The entry below measured why the proposal's ranking failed: dividing each
residual by the atom's *own* predicted scatter, and then paying
`- sum_d log sigma_jd` to normalise the density, makes vagueness cheaper than
closeness. This entry replaces that score with one shared metric and measures
the result. Diagnostic only; `--score production` remains the default and no
production path is touched.

**The change.** `common_metric_scale` takes one tolerance per coordinate from
the proposal cache alone: the catalogue's median predicted scatter for the
three linear coordinates, and its median *fractional* scatter, in dex, for
flux, which spans five decades and has no meaningful absolute tolerance. On
this cache that is `(0.342, 0.322, 1.239 pix, 0.337 dex)`. `common_metric_score`
then scores `log Pdet_j - 0.5 * squared distance in that metric`. Because the
metric no longer varies by atom, the normalisation is a constant and cancels in
the softmax, so a wide prediction can no longer buy a cheap residual. The
mixture composition, the temperature, `delta`, the exact stratum and the
fixed-count sampler are all unchanged; only the score differs.

**The ranking now tracks the observation.** Mean predicted shape by rank band,
against the observed value. Production's deeper bands were a constant
independent of the observation; they now move with it:

| | row 142230 | row 409188 | row 3563 |
|---|---:|---:|---:|
| observed `g1` | -0.2267 | -0.1213 | -0.0014 |
| ranks 1-1024, production | -0.0044 | -0.0090 | +0.0067 |
| ranks 1-1024, shared metric | **-0.0391** | **-0.0683** | **-0.0011** |
| ranks 1025-8192, production | +0.0062 | +0.0063 | +0.0065 |
| ranks 1025-8192, shared metric | **-0.0195** | **-0.0276** | **+0.0012** |

In `g2`, observed -0.0591 / +0.0970 / +0.1612 against ranks 1-1024 at
-0.0183 / +0.0591 / +0.0797, where production gave -0.0062 / +0.0068 / +0.0124.
The leading band now recovers 17%, 61% and 49% of the observation's own
displacement from the catalogue centre, against 2%, 7% and 8% before, and the
signature that mattered -- the same number whatever was observed -- is gone at
every depth.

**Reached mass, fixed-count sampler, same seed, same exact stratum:**

| row | production score | shared metric |
|---|---:|---:|
| 142230 | 12 of 32, 17.2% | **22 of 32, 87.5%** |
| 409188 | 27 of 32, 65.4% | **27 of 32, 92.8%** |
| 3563 | 25 of 32, 92.0% | 22 of 32, 87.8% |

**Row 3563 gets slightly worse, and the reason is structural.** The shared
metric spreads the proposal: the exact stratum's share falls from 87.7/89.6/89.2%
to 16.5/10.0/6.6%. On row 3563 the old, extremely peaked proposal already had
25 of the 32 mass atoms inside the deterministic top-1024; the flatter one has
17 there and the draw recovers 5, for 22. The row where the peaked proposal was
already right pays a little; the row where it was wrong gains 70 points.

**Spreading the proposal did not cost variance where it matters.** The worst
Horvitz-Thompson weight carried by a drawn mass atom is 2x, 2x and 6x on the
three rows, because the metric now places real probability on the atoms that
carry mass, so they are drawn with inclusion near one rather than stumbled on.
For comparison the production score leaves row 142230's heaviest missed atom,
0.3290, at the defensive floor with inclusion 3.4e-4 and a weight of 2,930x if
it is ever hit. Mass reached deterministically versus by draw is 0.549/0.270 on
row 142230, 0.569/0.091 on row 409188 and 0.590/0.035 on row 3563.

Files. `scripts/plot_proposal_atom_map.py` gains `common_metric_scale`,
`common_metric_score`, `mixture_from_score` and a `--score` flag, plus a
`score` block in the report. `scripts/run_proposal_atom_map.sh` takes the score
as its second argument.

Validation. `tests/test_proposal_atom_map.py`, 61 passed in 6.6s on the login
node and 12.3s inside the job; eleven new tests cover the scale, the score and
the mixture. One pins the defect directly: two atoms where the first is closer
to the observation in *every* coordinate, and production rates it 22 nats worse
because its narrow errors charge it for the miss. `mixture_from_score` is
tested to reproduce `WholeCatalogueProxy.mixture` bit for bit when fed the
production score, so the two paths cannot drift. Job 16627718 raised an
overflow in the fractional-scatter median where a few atoms carry a denormal
predicted flux; taking the median in log space is identically the same number
without the overflow, and job 16627763 reproduces 16627718's reached masses
exactly (0.8184, 0.6247, 0.6597), which confirms the fix changed nothing but
the warning. 16627763 COMPLETED in 00:02:42 on `cluster`, 8 CPUs, 96G, no GPU,
MaxRSS 4.1G.

Limitations. Three rows, centre node, one seed, so the reach numbers carry no
uncertainty; a seed sweep is needed before any of this is a population claim.
No estimator variance is measured -- the weight figures above are inclusion
arithmetic on the mass atoms, not a measured variance. The shared shape
tolerance, 0.342, is as wide as the whole ellipticity distribution because it
is a catalogue median dominated by faint atoms, which is why shape is still
only partly tracked while flux and size are tracked well; that single number is
the obvious next lever and was deliberately not tuned here. The metric is fixed
across observations, so a bright row and a faint row are judged on the same
tolerance.

Next steps. Make the shape tolerance reflect the observation's own brightness
rather than the catalogue median, which should close the remaining shape gap
without touching the rest. Re-examine whether the exact stratum should be
larger now that the proposal is flatter, since 1024 was chosen against a
proposal that put 88% of its mass there and now holds 7-17%. The nine-variant
comparison still must be redone on inclusion probabilities with the exact
stratum excluded.

## 2026-09-21 — The atom map hid the exact stratum, and the ranking is blind rather than offset

Owner looked at the fixed-count figures and objected that more of the ranked
draws should sit on the observation, since that is what the ranking is for, and
asked whether the ranking is instead simply offset in the `e1`-`e2` plane. Two
separate things were wrong, one in the figure and one in the proposal.

**The figure omitted most of the proposal.** `scripts/plot_proposal_atom_map.py`
coloured only the atoms the race drew. The exact stratum — the top 1024 by
proposal probability, summed with certainty and zeroed before the race — was in
neither the drawn set nor, except by accident of subsampling, the grey bulk. On
these rows that stratum holds 87.7%, 89.6% and 89.2% of the whole proposal, so
the figure was showing the leftovers and hiding the atoms the ranking most
prefers. It is now drawn as its own orange class, and the background subsample
excludes it so nothing is plotted twice.

**The ranking is not offset; past the leading atoms it is blind.** A new
`rank_bands` block in the report gives the mean predicted-minus-measured offset
per band of proposal rank. Mean *predicted* shape, recovered as observed plus
that offset:

| band | row 142230 | row 409188 | row 3563 |
|---|---:|---:|---:|
| observed `g1` | -0.2267 | -0.1213 | -0.0014 |
| ranks 1-1024, unweighted | -0.0044 | -0.0090 | +0.0067 |
| ranks 1-1024, q-weighted | **-0.2448** | **-0.1142** | **-0.0202** |
| ranks 1025-8192 | +0.0062 | +0.0063 | +0.0065 |
| ranks 8193-65536 | +0.0097 | +0.0093 | +0.0093 |
| whole catalogue | +0.0059 | +0.0058 | +0.0058 |

and in `g2` the same pattern: observed -0.0591 / +0.0970 / +0.1612 against
-0.0055 / -0.0056 / -0.0058 for ranks 1025-8192.

Past rank ~1024 the mean predicted shape is a *constant* to four decimals,
equal to the catalogue mean, across three observations that span 0.23 in `g1`
and 0.22 in `g2`. It is not a shift of the ranking towards a wrong centre: the
band carries no information about the observation at all. The apparent offset
in the figure is exactly minus the observation's own distance from the
catalogue centre, which is why it looked largest on row 142230 and why row
3563, observed at `g1 = -0.0014`, shows no `g1` offset while showing a large
`g2` one. Only the probability weighting recovers the observation, and it does
so because one atom carries 0.742 of the proposal on row 142230; the other
1023 members of the exact stratum are already catalogue-typical by count.

**Why it goes blind: vagueness is cheaper than being wrong.** The score is
`log Pdet - sum_d log sigma_jd - 0.5 sum_d z_jd^2` with each atom's own sigma.
Median terms on row 142230:

| group | `log Pdet` | `-sum log sigma` | quadratic | sigma(g1) | sigma(flux) |
|---|---:|---:|---:|---:|---:|
| drawn from ranked atoms | -0.41 | -12.27 | **-0.80** | 0.418 | 2.53e5 |
| mass-carrying atoms (exact) | -0.02 | -5.44 | **-7.51** | 0.068 | 2.27e4 |

The observed flux is 1.20e5, so a drawn atom's predicted flux uncertainty is
about twice the observation itself, and its `sigma(g1) = 0.42` spans the whole
ellipticity range. Such an atom fits *any* observation at under one sigma, so
its quadratic term costs 0.80 while a genuinely matching atom pays 7.51 for
missing inside its own narrow sigma. The dispersion term claws back 6.8 nats
of that, leaving the two groups within 0.1 nats of each other — and there are
twenty million vague atoms against thirty-two real ones. This is the same
defect the faint/small blue cloud shows in truth space, measured at its source,
and it is the case for the common-metric score variant that was already next in
the queue.

Files. `scripts/plot_proposal_atom_map.py`: the exact stratum is carried
through to all four panels and the legend, excluded from the background
subsample, and a `rank_bands` block is added to the report. Behaviour of the
sampler, the draw, and every reached-mass number is unchanged — job 16627577
reproduces job 16627124 exactly (12/32, 25/32, 27/32 atoms reached).

Validation. `tests/test_proposal_atom_map.py`, 50 passed in 6.7s on the login
node and 11.2s inside the job. Job 16627577 COMPLETED in 00:01:59 on `cluster`,
8 CPUs, 96G, no GPU, MaxRSS 5.2G. Figures refreshed in `SBSI/plots/`, which is
not version-controlled.

Limitations. Three rows, centre node, one seed; the band means are unweighted
over millions of atoms so their standard errors are negligible, but the claim
"constant across observations" rests on three observations, not a population.
No estimator variance is measured. The dispersion/quadratic trade is quantified
by medians only, at the median atom of each group.

Next steps. The common-metric score variant moves to first, now that the
mechanism is measured rather than inferred: score with a shared per-coordinate
metric so a wide predicted sigma cannot buy a cheap quadratic term. Enlarging
the exact stratum remains the cheap deterministic lever and still needs the
rank of the missed tail atoms, which `rank_bands` does not yet report per atom.
The nine-variant comparison still must be redone on inclusion probabilities
with the exact stratum excluded.

## 2026-09-21 — A fixed-count stratified draw finds one more mass atom per row, and confirms the ranking is the real defect

Owner asked for the atom map redrawn with a 50/50 split on *draw counts* rather
than on probability, having noted that the current sampler looks inefficient.
That is a different design from the `delta = 0.5` mixture weight ruled out in
the entry below, and unlike that one it is worth something.

**What the current sampler does.** Production races one mixture, so the split
of the 16,384 slots between the ranked and flat components is not chosen: it
falls out of their mass ratio once the exact stratum is removed. On row 142230
that is about 3,100 ranked against 13,300 uniform. The ranked component never
asked for so few slots; it simply holds little mass after the top-1024 are
taken out and summed exactly.

**The alternative implemented here.** `stratified_race` in
`scripts/plot_proposal_atom_map.py` runs two independent without-replacement
draws over the atoms the exact stratum leaves behind:

- 8,192 slots raced on the softmax tail alone (`q_j - floor`), so the race sees
  the ranking's preference and nothing else. Its threshold falls as its budget
  rises, which is the point.
- 8,192 slots at equal weight, which makes the same race a simple random sample
  without replacement, so every eligible atom is included with exactly
  `n_uniform / n_eligible`.

An atom may win in both and is credited to the ranking when it does; combined
inclusion is `1 - (1 - i_r)(1 - i_u)`, which is what the Horvitz-Thompson
weight divides by. If the ranking prefers fewer atoms than it has slots the
leftovers are handed to the uniform stratum rather than left unspent, so the
budget is never quietly shrunk. Selected with `--sampler fifty-fifty`;
`production` remains the default and is unchanged.

Measured, centre node, exact stratum still 1024, same seed:

| row | ranked tau | uniform inclusion | mass atoms reached | reached mass | was |
|---|---:|---:|---:|---:|---:|
| 142230 | 2.826e-6 | 3.413e-4 | 12 of 32 (11 exact + 1 drawn) | 17.2% | 16.0% |
| 409188 | 4.980e-7 | 3.413e-4 | 27 of 32 (26 exact + 1 drawn) | 65.4% | 62.9% |
| 3563 | 9.705e-7 | 3.413e-4 | 25 of 32 (25 exact + 0 drawn) | 92.0% | 92.0% |

No atom saturates in either stratum on any row; the two strata overlap on 1, 3
and 3 atoms.

Findings.

- **The random draw now finds something.** Under the production sampler the
  16,384 draws found zero mass-carrying atoms on all three rows; everything
  reached came from the deterministic top-1024. The fixed-count draw finds one
  on row 142230 (atom 6528493, mass 0.0117, inclusion 0.272 -> 0.725) and one
  on row 409188 (atom 21000023, mass 0.0176, inclusion 0.0598 -> 0.755). Small,
  but it is the first time this budget has contributed anything on these rows.
- **The realised gain matches the predicted gain.** Before running it, the
  expected recovered mass from the missed atoms was computed analytically as
  0.0131, 0.0162 and 0.0010 for rows 142230, 409188 and 3563, against realised
  0.0118, 0.0176 and 0.0000. The agreement is a check on the inclusion
  arithmetic, not a tuned result; nothing was adjusted after seeing it.
- **It is a second-order fix and the entry below says why.** Row 142230 still
  misses 20 atoms holding 0.774 of the captured mass, and the single heaviest,
  0.3290, sits at exactly the defensive floor with no softmax preference at
  all. The ranked stratum cannot see such an atom at any budget, and the
  uniform stratum now sees it *less* often than before: shrinking the uniform
  side from about 13,258 slots to 8,192 takes a floor atom's inclusion from
  5.5e-4 to 3.4e-4, so its Horvitz-Thompson weight on a hit rises from about
  1,810x to 2,930x. The same holds for the 0.2002 atom on row 409188. The trade
  is a real one and it is favourable only because the tail holds more of the
  missed mass than the floor does: 62%, 88% and 67% on the three rows.
- **The figures show the ranking defect unchanged.** With 8,192 ranked draws
  instead of about 3,400 the blue cloud is far denser but sits in the same
  place: the faint, small corner of truth space (true `r` about 26-28,
  `R_e` about 0.1 arcsec) and about one decade low in predicted flux, for rows
  whose observation is at `r = 17.3` and `18.35`. More slots spent on a
  ranking that prefers the wrong atoms buys proportionally more wrong atoms.

Validation. `tests/test_proposal_atom_map.py`, 50 passed in 6.6s on the login
node and 15.3s inside the job; eight new tests cover the fixed-count draw
(slot counts, exclusion of the exact stratum, the uniform inclusion identity,
the leftover-slot reallocation, monotonicity of the ranked threshold in its
budget, double-winner attribution, independent combination of the two
inclusions, and two rejection paths). The reallocation path was added because
the first implementation raised `priority threshold not positive finite` when
fewer atoms carried softmax mass than the ranked stratum had slots, which five
tests caught. Job 16627124 COMPLETED in 00:02:12 on `cluster`, 8 CPUs, 96G, no
GPU, MaxRSS 5.7G. Figures copied to `SBSI/plots/` as
`proposal_atom_map_row*_fifty_fifty.png`, which is not version-controlled.
`scripts/run_proposal_atom_map.sh` is added because the equivalent launcher for
the earlier jobs in this series was never committed and had to be
reconstructed. Diagnostic only: the fixed-count sampler is not wired into any
production path, and nothing here changes target, model, cuts, prior or
production settings.

Limitations. Three rows from the worst-curvature list, centre node only, one
seed, so "one more atom per row" carries no uncertainty and should not be read
as a population result; a seed sweep would be needed to attach an error bar to
the reached-mass change. The 50/50 split itself is the owner's proposal, not an
optimum: the same arithmetic gives 0.0184 and 0.0214 expected recovered mass at
12,288 ranked / 4,096 uniform, at the cost of taking floor-atom inclusion down
to 1.7e-4. No estimator variance is measured here, only inclusion
probabilities and reached mass.

Next steps. Unchanged in priority. The ranking, not the budget split, is what
fails on these rows, so the common-metric score variant remains first. If a
sampler change is wanted before that, enlarging the exact stratum is the
cheaper and safer lever than reallocating random slots, because it is
deterministic; that needs the rank of the missed tail atoms measured first,
which this diagnostic does not yet report. The nine-variant comparison still
must be redone on inclusion probabilities with the exact stratum excluded.
# Work log

## 2026-09-21 — Correction: the exact stratum does all the work, and delta cannot change that

Owner asked whether the 0.9/0.1 split between the ranked and flat components is
a good choice, and suggested 50/50. Answering it required fixing a defect in
this diagnostic, which also retracts a claim made in the entry below.

**The diagnostic was reproducing a sampler nobody runs.** The run's
`pipeline_config.proposal.candidates` is 1024: production takes the top 1024
atoms by proposal probability, sums them exactly, and zeroes their weight
before the priority race, so they consume no draw and contribute no variance
(`DefensiveLocalProposal.draw_priority` passes them as `excluded`). This script
raced them. `PRODUCTION_CANDIDATES` now reproduces the exclusion.

**Retraction.** The entry below reports that the ranked 90% saturates on tens
of atoms and wastes ~113,000 slots' worth of probability. That is an artifact
of racing the exact stratum. With the stratum excluded, **no atom saturates on
any of the three rows**, and the ranked tail converts its mass exactly
linearly: 3,092 slots predicted against 3,093 if it were spread thin. The
saturation argument, and the reading of tempering as "making the 0.9
spendable", do not survive. The measured split is not waste; it is the mass
ratio of what remains after the head is removed.

Measured, centre node, exact stratum excluded:

| row | exact stratum mass | ranked tail mass | tau | flat slots | ranked-tail slots | saturated |
|---|---:|---:|---:|---:|---:|---:|
| 142230 | 0.8767 | 0.0233 | 7.543e-6 | 13,258 | 3,092 | 0 |
| 409188 | 0.8960 | 0.0040 | 6.354e-6 | 15,738 | 627 | 0 |
| 3563 | 0.8919 | 0.0081 | 6.636e-6 | 15,068 | 1,220 | 0 |

**The finding that replaces it, and it is worse.** Counting the exact stratum
as a reach, which it is:

| row | mass atoms reached | summed exactly | found by the 16,384 draws | mass reached |
|---|---:|---:|---:|---:|
| 142230 | 11 of 32 | 11 | 0 | 16.0% |
| 409188 | 26 of 32 | 26 | 0 | 62.9% |
| 3563 | 25 of 32 | 25 | 0 | 92.0% |

The random draw found no mass-carrying atom on any row. Everything reached came
from the deterministic top-1024. On row 142230 that leaves 78.6% of the
captured posterior on 21 atoms that are neither summed nor drawn, each with
inclusion probability 5.5e-4.

**Why delta is the wrong knob.** A floor atom is included with
`(delta/N)/tau`, and when the flat component dominates the race
`tau ~ delta/M`, so delta cancels. The ceiling at any delta is the budget over
the catalogue size, `M/N = 16384/24e6 = 0.068%`. Holding the stratum fixed and
using the measured tail masses:

| delta | floor-atom inclusion, row 142230 | ranked-tail slots |
|---|---:|---:|
| 0.1 (production) | 0.055% | 3,096 |
| 0.25 | 0.063% | 1,181 |
| 0.5 | 0.067% | 413 |
| 0.9 | 0.068% | 47 |
| 1.0 | 0.068% | 0 |

50/50 would raise a heavy floor atom's chance from 0.055% to 0.067% while
cutting the ranked tail's slots by 87%. Spending the entire budget uniformly
caps at 0.068%. No delta reaches a specific heavy atom in a 24m catalogue with
16k draws; delta only divides a budget that is too small under either split.
This is an argument that delta cannot help on these rows, not a measurement of
the best delta for the population, and it is analytic rather than a simulated
sweep.

**Where the leverage is.** The exact stratum is doing 100% of the work at 1024
atoms. Enlarging it helps only if the heavy atoms rank anywhere near it: on row
142230 the two heaviest sit at the defensive floor, below roughly 1.2m other
atoms, so no plausible stratum size reaches them. Only fixing the ranking does,
which points back at the wide-predicted-scatter defect in the entry below.

Validation. `tests/test_proposal_atom_map.py`, 41 passed in 6.6s on the login
node and 25.6s inside the job. Job 16626863 COMPLETED in 00:02:23 on `cluster`,
no GPU. The figure headline and legend now separate "summed exactly" from
"drawn", and the report carries an `exact_stratum` block and per-atom
`inclusion_probability` and `summed_exactly` flags. Figures and reports copied
to `SBSI/plots/`, which is not version-controlled. Diagnostic only: no flow
evaluation, no change to target, model, cuts, prior or production settings.

Limitations. Three rows from the worst-curvature list, centre node only, one
seed. The delta table holds the exact stratum membership fixed, which is exact
here because the top-K by `q` is the top-K by softmax score whatever delta is,
but it assumes the budget and stratum size stay at 16384 and 1024. Nothing
here measures estimator variance; the ordering of proposals still needs the
inclusion-probability rework recorded below.

Next steps. Unchanged in substance, but reprioritised: the proposal's ranking,
not its defensive share, is what fails on the bright blended rows, so the
common-metric score variant is now the first thing to test. The nine-variant
comparison must be redone on inclusion probabilities before any of its numbers
are used, and it must also exclude the exact stratum, which it does not.

## 2026-09-21 — The ranked 90% cannot spend its mass: priority sampling caps every atom at one slot

Owner asked why, if the mixture puts 90% of its probability on the ranked
component, 76% of the draws land on floor atoms. The two numbers are both
right and are not in conflict: 90% is a share of *probability*, and shares of
probability equal shares of *draws* only with replacement. Production draws
without replacement.

Priority sampling includes atom `j` with probability `min(1, q_j / tau)`.
The `min` is the whole story: probability an atom holds above `tau` buys
nothing, because the atom is drawn at most once. A thin component converts its
mass into slots linearly, at `mass / tau`; a concentrated one saturates and
converts almost none of it.

Added `slot_accounting` and made `priority_race` return the threshold it
already computed. Measured, centre node, 16384 draws:

| row | tau | flat 10%: mass -> slots | ranked 90%: mass -> slots | ranked slots if spread thin | saturated atoms | max `q` |
|---|---:|---:|---:|---:|---:|---:|
| 142230 | 7.723e-6 | 0.100 -> 12,948 | 0.900 -> 3,411 | 116,532 | 38 | 0.742 |
| 409188 | 6.454e-6 | 0.100 -> 15,495 | 0.900 -> 881 | 139,453 | 131 | 0.508 |
| 3563 | 6.765e-6 | 0.100 -> 14,781 | 0.900 -> 1,512 | 133,029 | 126 | 0.745 |

The predicted split matches the realised draw: row 142230 predicts 12,948
uniform and 3,411 ranked against 12,998 and 3,386 actually drawn, and the
inclusion probabilities sum to 16,359 against a budget of 16,384.

Findings.

- The ranked component is not getting 90% of the draws; it is getting 3,411 of
  16,384 on row 142230 and 881 on row 409188. Had its 0.9 been spread thinly
  enough to stay under `tau` it would have claimed 116,532 slots, seven times
  the entire budget. Instead it saturates on 38 atoms and the rest evaporates.
- The flat 10% is not being generous; it is the only component thin enough to
  convert mass into slots linearly, so it absorbs every slot the ranking cannot
  use. The 76% figure measures the ranking's waste, not the defence's reach.
- This is the mechanism behind the tempering result in the proposal-coordinate
  entry. `T = 2` helps not by ranking atoms better but by flattening the
  softmax so more of its 0.9 sits below `tau` and becomes spendable.

Legend simplified at the owner's request to three entries -- not drawn, drawn
uniformly (the flat 10%), drawn from ranked atoms (the 90%) -- and
`draw_classes` reduced to two classes accordingly, cut at `q > 2 * floor`,
the point where the ranking supplies more of an atom's probability than the
flat spread does. The previous three-class split is superseded; its
`preferred`/`weak` boundary at the flat share is retained in the report as
`n_above_flat_share_in_catalogue` only.

Validation. `tests/test_proposal_atom_map.py`, 41 passed in 6.6s on the login
node and 7.5s inside the job. The slot-accounting test caught a real bug in the
first implementation: it credited a saturated atom a flat share *in addition*
to its capped single slot, so the two parts did not sum to the realised draws.
The decomposition now credits a saturated atom's slot entirely to the ranking,
and a test asserts the parts sum exactly. Job 16626722 COMPLETED in 00:01:47 on
`cluster`, no GPU; supersedes 16626668, 16626281 and 16626236. Figures and
reports copied to the owner's checkout at `SBSI/plots/`, which is not
version-controlled. Diagnostic only: no flow evaluation, no change to target,
model, cuts, prior or production settings.

Limitations. `tau` and the slot split are exact for this seed and this row;
they are not averaged over seeds and carry no uncertainty. Three rows from the
worst-curvature list, centre node only. The accounting describes how the budget
is spent, not whether spending it differently would lower the variance of the
estimator -- that still needs the inclusion-probability rework noted below.

Next steps unchanged: redo the nine-variant comparison on inclusion
probabilities `min(1, q/tau)` rather than the with-replacement second moment
`S`, which now has a second reason to matter -- `S` cannot see saturation
at all, and saturation is where this proposal loses its budget. Then test the
common-metric score proposed in the entry below.

## 2026-09-21 — Why the proposal prefers vague atoms: wide predicted scatter buys compatibility

Owner looked at the atom map and asked two questions: why the drawn atoms are
not centred on the observation in the predicted `g1`/`g2` plane when the score
is built to centre them, and why in truth space the draw piles up in the faint,
small corner. Both are answered by the same measurement, and the answer is a
defect in the proxy score, not in the figure.

First, a labelling error in the previous entry is corrected. The class drawn
in blue and called "ranked by the proxy" was defined as `q > delta / n_atoms`.
That floor is reached only when the softmax underflows in float64, about 745
nats below the best atom, so the test is satisfied by nearly the whole
catalogue and does not mean the proposal prefers the atom. `draw_classes` now
cuts against the flat share `1 / n_atoms` and returns three classes:
`preferred` (`q > 1/N`), `weak` (scored but at or below the flat share) and
`at_floor`. Counts, centre node:

| row | preferred | weak | at floor | above the flat share in 24m |
|---|---:|---:|---:|---:|
| 142230 | 3,175 | 805 | 12,404 | 71,823 |
| 409188 | 720 | 1,108 | 14,556 | 28,615 |
| 3563 | 1,344 | 1,110 | 13,930 | 43,063 |

So the correction does not rescue the picture: on row 142230, 3,175 of the
3,980 non-floor draws are atoms the tilt genuinely prefers. The offset the
owner saw is a property of the proposal.

`score_terms` and `summarise_terms` now split the proxy score into its three
additive parts and report, per coordinate, the median predicted scatter, the
median raw miss and the median standardized miss. Row 142230, centre node:

| group | median score | median `-sum log sigma` | median `-0.5 sum z^2` | sigma(g1) | \|resid\|(g1) | \|z\|(g1) | sigma(flux) | \|resid\|(flux) | \|z\|(flux) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top 64 by `q` | -11.17 | -8.81 | -1.72 | 0.156 | 0.140 | 0.76 | 1.11e5 | 8.28e4 | 0.70 |
| drawn, preferred | -13.79 | -12.03 | -0.87 | 0.415 | 0.235 | 0.57 | 2.27e5 | 9.92e4 | 0.43 |
| mass-carrying (exact) | -15.95 | -5.44 | -7.51 | 0.068 | 0.119 | 1.34 | 2.27e4 | 3.54e4 | 1.88 |

Findings.

- The preferred atoms are centred, in the units the score actually uses: their
  median standardized miss is 0.57, 0.15, 0.95 and 0.43 sigma across `g1`,
  `g2`, `R_flux` and flux. They look off-centre in the figure because the
  figure is in raw units and the score is in units of each atom's own
  predicted scatter.
- They are *further* from the observation than the mass-carrying atoms in raw
  units in three of the four coordinates (`g1` 0.235 against 0.119, `R_flux`
  4.57 against 1.29, flux 9.9e4 against 3.5e4), and win anyway because their
  predicted scatter is 6x wider in `g1` and 10x wider in flux. The `-sum log
  sigma` penalty does not cover the gap: the preferred atoms pay 6.6 nats more
  in scatter and gain 6.6 nats in fit, and the residual 2 nats tip the ranking
  the wrong way. Wide predicted scatter buys compatibility.
- The faint, small corner in truth space is the same population. Those atoms
  are true mag ~27 at `Re` ~0.1 arcsec, yet panel 1b puts their predicted flux
  only about one dex below the observation rather than the four dex their own
  brightness implies. They are faint galaxies whose predicted measurement is
  dragged up by a bright neighbour, with correspondingly large predicted
  scatter. The proposal prefers them for that scatter.
- The contrast with a working row is sharp. On row 3563 the mass-carrying
  atoms are the top-scoring atoms: median score -7.66 against -7.28 for the
  top 64, with near-identical scatter (sigma(g1) 0.090 against 0.086,
  sigma(flux) 8.7e3 against 9.5e3), and both sit well above the preferred bulk
  at -12.16. On row 142230 the mass atoms score *below* the generic
  above-uniform bulk, -15.95 against -15.51. The proposal is not merely
  under-resolving the right answers on the bright blended object; it ranks
  them below a cloud of vague ones.
- The floor class is a numerical underflow, not a judgement: its median
  standardized flux miss is 3242 sigma on row 142230, because a faint isolated
  atom predicts a flux near zero with a scatter of ~37 against an observation
  of 1.2e5.

Two readings of the last point, not separated by this work. Either the flow's
predicted mean for the correct atoms is genuinely off by 1.3-1.9 sigma on this
object, which would also put the likelihood off-centre, or the proxy's
*diagonal* Gaussian is a poor surrogate for the flow likelihood precisely
where blending correlates flux, size and shape, in which case only the
proposal is affected. The exact posterior that defines the mass atoms comes
from the flow, not from the proxy, so the two cannot be told apart from these
numbers alone. Separating them is the next question, and it decides whether
this is a proposal-tuning problem or a likelihood problem.

Suggested next variant, on the existing compare harness rather than in
production: score with a common per-coordinate metric instead of each atom's
own sigma, which turns the density into a distance and removes the reward for
vagueness. On row 142230 the mass atoms are closer than the preferred atoms in
three of four raw coordinates, so a common metric would rank them above; this
is a prediction the harness can falsify.

Validation. `tests/test_proposal_atom_map.py`, 39 passed in 6.8s on the login
node and 14.9s inside the job, including a test pinning `score_terms` against
the score formula in `WholeCatalogueProxy`'s own docstring, one showing a
narrower sigma raising the score at an equal standardized miss, one confirming
the old `q > floor` rule mislabels a near-floor atom as preferred, and one
checking an undetected atom's `-inf` does not poison a summary. Jobs 16626236
and 16626281 COMPLETED in 00:02:01 and 00:01:53 on `cluster`, no GPU; 16626281
supersedes it and is the copy in `SBSI/plots/`. Diagnostic only: no flow
evaluation, no change to target, model, cuts, prior or production settings.

Limitations. Three rows from the worst-curvature list, centre node only, and
the mass-carrying set is the 32 atoms the exact calculation retained, so
"mass-carrying" means "retained and heavy", not "all the mass". The medians
are unweighted over atoms and carry no uncertainty; with n=32 for the mass
group the quoted medians are indicative, not measured to a stated precision.
No spread is reported on any median here, so none of these differences has a
significance attached.

## 2026-09-21 — Atom map: where the mass is versus where the proposal looks

Owner asked for a picture of the prior atoms of one observation, on their
predicted measurements centred on the observation and on their true
properties, with drawn atoms marked separately from undrawn ones and coloured
by posterior mass. Added `scripts/plot_proposal_atom_map.py`,
`jobs/job_plot_proposal_atom_map.sh` and `tests/test_proposal_atom_map.py`.

Diagnostic only. No flow was evaluated and no GPU was requested: the figure
reuses the cached proposal coordinate table, the cached zero-shear detection
probabilities and the exact 24m centre-node posterior weights already on disk,
so it does not consume the owner's two-GPU limit. The target, model, cuts and
prior are untouched.

The draw is reproduced from `DefensiveLocalProposal.select_priority_batch`:
key `q_j / u_j`, generator seeded by `_priority_row_seed(8701, row)`, keep the
16384 largest. The tilted-stratified exact stratum is not excluded, so the
drawn set is the proposal's own race rather than the estimator's final
bookkeeping.

Results, centre node, worst-curvature rows:

| row | measured mag | atoms drawn | captured mass reached | at defensive floor | mass at floor | mass rank of the proposal's top atom |
|---|---|---|---|---|---|---|
| 142230 | 17.30 | 9/32 | 15.1% | 9 | 32.6% | 3 |
| 409188 | 18.35 | 23/32 | 53.8% | 3 | 32.5% | 3 |
| 3563 | 18.45 | 24/32 | 89.2% | 3 | 2.6% | 2 |

Findings.

- The failure is a hard priority threshold, not wasted duplicate draws. For
  row 142230 the race cuts cleanly: every atom with `q >= 7.6e-6` was drawn
  and every atom with `q <= 4.9e-6` was not. No floor atom was drawn, which is
  what 16384 draws from 24m predicts.
- Row 142230's two heaviest atoms, 51.0% of the posterior between them, both
  sit at the defensive floor `4.1667e-09`, tied with the 24m atoms the
  proposal treats as impossible. Its single favourite atom, `q = 0.742`, is
  only the third heaviest and carries 8.6%.
- The owner's blending argument is visible in the figure. On row 142230 the
  mass-carrying atoms scatter over roughly +-0.3 in predicted `g1`/`g2`
  relative to the observation, while the same atoms sit in a tight clump near
  zero in predicted size and flux. Intrinsic shape shows no concentration at
  all, and drawn versus missed atoms are interleaved in it.
- Atoms that matter are bright and large (mag 16.5-18.5, `Re` 0.9-2.9 arcsec)
  and sit in a sparsely populated corner of a prior whose bulk is mag 24-28 at
  `Re < 0.5`.
- Counting atoms found is misleading. Row 409188 drew 23 of 32 yet reached
  only 53.8% of the mass, because the atoms it missed were the heaviest and
  the largest in `Re`.
- Row 3563, a magnitude fainter and less blended, behaves well on the same
  machinery: mass-carrying atoms cluster around the observation in predicted
  shape and 89.2% of the mass is reached.

Correction to the 2026-09-21 proposal-coordinate entry. The production draw is
priority sampling without replacement, so a dominant `q` takes one slot, not a
proportional share of the 16384 draws. `S = sum_j w_j^2 / q_j` as reported
there is the with-replacement second moment; the priority estimator's variance
runs on inclusion probabilities `min(1, q_j / tau)`. The variant ordering is
expected to be unaffected, since tempering helps by lifting starved atoms over
`tau`, but the absolute ESS figures in that entry describe a sampler that is
not the one in production.

Revision, same day, at the owner's request. Three changes to the figure and
one new measurement.

- The observation's own true properties are now plotted, as a dashed crosshair
  in both truth panels. They are recoverable after all: the case input feather
  named in `input/image_mock_manifest.json` `per_case[].sources.truth.path`,
  indexed by `source_input_index` and checked against its `index_input`
  column. Intrinsic ellipticity and circularized radius are rebuilt from axis
  ratio and position angle with the atom convention, `|e| = (1-q)/(1+q)` and
  `Re * sqrt(q)`, and agree with the stored atom columns to 1e-8. Values:
  142230 `e = (-0.2660, -0.0657)`, `Re_c = 1.446"`, `r = 17.275`, case 114;
  409188 `e = (-0.1508, +0.1036)`, `Re_c = 1.586"`, `r = 18.329`, case 58;
  3563 `e = (-0.0214, +0.1610)`, `Re_c = 1.168"`, `r = 18.335`, case 113.
  In the size/brightness truth panel the crosshair sits among the
  mass-carrying atoms, so those atoms are not off in an implausible corner.
- Drawn atoms are now split by why they were drawn. An atom whose mixture
  probability is at the defensive floor `delta/24m` won the race on its
  uniform share alone; anything above the floor was ranked by the proxy. They
  are drawn in two colours at their true counts.
- Every undrawn atom, background or mass-carrying, is now the same small dot,
  so size no longer codes for anything but "drawn". Legibility is carried by
  colour, opacity and a hairline edge instead.

New measurement, the draw-class split:

| row | drawn, ranked by the proxy | drawn on the uniform floor | atoms above the floor in 24m |
|---|---:|---:|---:|
| 142230 | 3,980 | 12,404 | 1,231,271 |
| 409188 | 1,828 | 14,556 | 1,435,576 |
| 3563 | 2,454 | 13,930 | 1,540,103 |

Most of the draw is the defensive lottery, not the proposal: 76%, 89% and 85%
of the 16384 slots go to atoms the proxy scored at the floor. The proxy is
effectively spending only a few thousand slots on its own ranking, which is
consistent with the concentration defect reported above and means the 0.1
defensive fraction is doing more of the work than its name suggests. This is a
count, not a variance statement; it does not by itself say the floor draws are
wasted.

Validation. `tests/test_proposal_atom_map.py`, 33 passed inside the job. Job
16625903 COMPLETED in 00:01:34, partition `cluster`, no GPU; supersedes
16625836 and 16623681, which produced the same numbers with the earlier
figure. Two earlier attempts failed and were repaired: 16623593 on the
detection cache being point-major `(10, 24000000)` rather than atom-major, and
16623655 on `exact/row` being present in only 2 of the 32 panel records. Both
failure modes are now covered by unit tests. Figures and machine-readable
reports copied to the owner's checkout at `SBSI/plots/proposal_atom_map_row*.png` and
`SBSI/plots/proposal_atom_map_row*_report.json` (that directory is not tracked in git).

Limitations. Mass is known only for the 32 atoms the exact calculation
retained per node, capturing 93.5%, 71.2% and 71.1% of the posterior for the
three rows; the remainder is spread somewhere across the grey background,
which is a uniform 60k subsample and is not proven massless. The three rows
are the top of the worst-curvature list and are not representative. Only the
centre stencil node is shown. The background is a uniform subsample, so a
drawn atom in a sparse region may be a subsample artifact rather than a
neighbour of the observation.

Next steps. Redo the nine-variant comparison on priority-sampling inclusion
probabilities rather than `S`, so the quoted efficiency matches the production
estimator. Quantify, rather than eyeball, the shape-versus-size split by
comparing standardized proposal distances in the two coordinate pairs for
drawn and missed atoms.

## 2026-09-21 — Proposal-coordinate comparison on the worst-32 exact panel

Owner asked whether the tilted proposal should rank atoms on magnitude and
radius alone instead of all four measured coordinates, reasoning that
anisotropic blending shifts the e1/e2 centre of the likelihood. Added
`scripts/compare_proposal_coordinates.py`,
`jobs/job_compare_proposal_coordinates.sh` and
`tests/test_proposal_coordinate_compare.py`.

Proposal-only diagnostic. `q` is the only object varied; the target, model,
cuts, prior weights and production settings are untouched, and every
contribution still carries the exact `pi_j / q_j` correction, so each variant
is an equally valid estimator of the same quantity. No flow was evaluated and
no GPU was requested: the run reuses the cached proposal coordinate table
(`disk_assembled_v1/proposal/coordinates.npz`) and the exact 24m centre-node
posterior weights in `tail_exact_16617850/`, so it does not consume the
owner's two-GPU limit.

The script drives the production `WholeCatalogueProxy` through a shim that
supplies exactly the five attributes it reads, and inverts the defensive blend
to recover the bare softmax so components can be mixed. Variants are convex
mixtures of local components, each with its own coordinate subset, dispersion
inflation and softmax temperature. Reproduction of the recorded production
proposal is asserted, not assumed: `prod_4d` matches the 32 shared atoms of
`proposal_probe_16617724/retrieval.json` to a maximum relative difference of
2.40e-13.

Metric is the importance-sampling second moment `S = sum_j w_j^2 / q_j` over
the 32 exact top atoms per row, a lower bound on the full-prior second moment
because the retained atoms capture 0.68--0.99 of the centre-node posterior.
Lower is better. Geometric mean over the panel, paired against production:

| variant | geom-mean S | median S | worst S | rows better |
| --- | ---: | ---: | ---: | ---: |
| prod_4d (production) | 5588 | 8336 | 3.50e7 | -- |
| rf_2d (radius+flux) | 9127 | 1.36e4 | 3.49e7 | 14/32 |
| prod_4d_T2 | 2449 | 1223 | 1.90e7 | 24/32 |
| prod_4d_T3 | 4707 | 2895 | 1.08e7 | 20/32 |
| prod_4d_T4 | 9295 | 4480 | 9.66e6 | 17/32 |
| rf_2d_T2 | 6438 | 4015 | 1.45e7 | 19/32 |
| mix_4d_rf2d | 5168 | 7513 | 3.49e7 | 16/32 |
| mix_4d_rf2d_T2 | 2725 | 1668 | 1.61e7 | 25/32 |
| shape_sigma_x3 | 6193 | 8551 | 3.46e7 | 15/32 |

Findings. First, the owner's mechanism is confirmed but is not where the gain
is. Dropping the shape coordinates helps exactly where production fails and
hurts where it works: the correlation between `log S_prod` and
`log(S_rf2d/S_prod)` is -0.541 (p=0.0014, n=32); the nine brightest rows
(MAG_AUTO<18.5) improve by a geometric-mean factor 0.88 while the other 23
degrade by 2.08. On its own `rf_2d` is worse on average.

Second, the dominant defect is proposal concentration, not coordinate choice.
Production puts a median 0.54 of its mass on a single atom, and for row 142230
one atom takes 0.742 while 12 of 33 important atoms sit at the defensive floor
`delta/24m`. Tempering the existing softmax addresses this directly: T=2 gives
a paired ratio 0.44 (68% interval 0.33--0.58, p=0.007) and raises the median
effective-sample-size bound at 16384 draws from 2.0 to 13.4.

Third, adding the two-dimensional component on top of tempering is not
established. `mix_4d_rf2d_T2` improves the most rows (25/32) and is best on
the bright subset, but against `prod_4d_T2` alone the paired ratio is 1.11
(p=0.181) over the panel and 0.87 (p=0.423) on the nine bright rows. With
n=9 that comparison is underpowered and neither direction is demonstrated.

Limitations. The worst-32 panel was selected by production negative-curvature
contribution, so it is neither random nor held out and the absolute S values
are not population numbers; only the ordering of variants on identical rows is
supported. S is a lower bound restricted to the retained atoms, and the
effective-sample-size figures are correspondingly upper bounds. The
comparison is at the existing centre only and says nothing about scores,
information, derivative accuracy or the separate domain/prior and
normalization gates. No temperature was adopted and no production setting was
changed.

Validation: 16 focused tests passed in 18.49s, including reproduction of an
explicit Gaussian mixture through the production proxy class, inversion of the
defensive blend, and convex-weight checks. Job 16623076 completed in 3m23s on
`cluster` (8 CPU, peak 7.5 GiB); superseded runs 16623034 (failed on an
attribute name, now covered by a test) and 16623045 are preserved. Ruff,
compilation, Bash syntax and `git diff --check` passed. Report:
`proposal_coordinate_compare_16623076/report.json`.

Next steps: bracket the temperature between 1.5 and 3 against scores and
information rather than the centre-node second moment alone; repeat on a
reserved uniform panel before any production change, since the present panel
cannot support an acceptance decision.

Note on repository state: the entire V3.6 disk pipeline was uncommitted in the
working tree when this work started (`dev` at 63b8102 predates
`catalogue_disk_likelihood.py`, `disk_inference_store.py`,
`output_conditioned_response.py` and `crowding.py`). Those files were verified
byte-identical to the implementation hashes recorded in
`row_142230_exact.json` and committed on this branch so the pipeline exists in
git; they are unchanged from the owner's working tree.


## 2026-09-21 — Consolidate completed plot diagnostics and current stopping point

Updated this worklog and `doc/V36_INFERENCE_REVIEW.md` after delivering the four
matched-truth corner plots. Removed stale pending-output/clarification wording
from the failed prior-mixture attempt, preserving its failure and provenance.
Added the completed matched-truth result to the review's opening decision and
linked the final figures and machine-readable report from its results section.

Current conclusion: four selected problem cases at MAG_AUTO 17.6253, 18.4570,
19.4945 and 20.5649 show no extreme individual marginal discrepancy when
conditioned on their own simulation truth/neighbours at actual shear (0.02, 0).
This is not a representative validation sample, a four-dimensional joint
calibration test, or a test of shear derivatives. The older prior-mixture slices
answer a different question and do not directly diagnose the own-truth flow.
No inference cuts, model artifacts, prior weights or production settings changed.
The 100k production launch remains withheld; no new computation was submitted
for this documentation update. Next scientific work remains the unresolved
prior/proposal, derivative, normalization and model-coverage checks described in
the [review's next decision](V36_INFERENCE_REVIEW.md#limitations-and-next-decision).

Validation: read the final JSON report and checked all four row identities,
draw counts and marginal CDFs against the recorded results; all eight final
PNG/PDF files exist. `git diff --check` passed. Documentation only; no tests
rerun (the generation job's five passing tests remain recorded below).

## 2026-09-21 — Own-truth conditional corners across MAG_AUTO17–21

Owner clarified that the requested plot conditions on each galaxy's actual
simulation truth, not a diagnostic prior-atom mixture. Added
`scripts/plot_disk_truth_corners.py`, `jobs/job_plot_disk_truth_corners.sh`, and
`tests/test_disk_truth_corners.py`. Frozen rows239507/440648/274111/10642 have
MAG_AUTO17.6253/18.4570/19.4945/20.5649: nearest each one-magnitude-bin midpoint
among the existing worst50 approximate-production curvature contributors,
chosen before evaluating predictions. They are illustrative development cases.

The script verifies source hashes, frozen row identities, truth/detection joins,
actual measurements, and intrinsic morphology against generated catalogues.
It builds each primary's eight flow features from its rendered full scene,
explicitly converts degree angles, applies actual shear(.02,0) once to intrinsic
shape, samples the retained flow in FP64, and composes the output-conditioned
disk response using the nearest20 nonself neighbours within10arcsec. Crowding
uses all rendered neighbours within7arcsec, without truth cuts. At fixed truth
the usable-event classifier is a scalar and cancels from the law conditional
on usability; no prior mixture or measured analysis cuts are used here.

262144 draws per object yield 1D marginal histograms and 2D histogram-based
68/95% contours. No KDE, model tuning or clipping. Invalid numerical draws are
counted explicitly before/after response and excluded from the displayed valid
sample law; their frequency is retained on each figure and in JSON. Samples,
contexts, pair IDs/distances, source hashes and marginal CDFs are saved.
Matplotlib/resource-check skills informed layout and the oneA40/fourCPU/64GiB
scheduled allocation; user queue was empty before submission. Ruff, compilation,
Bash syntax and `git diff --check` passed. Job16622674 completed in1m16s,
with five focused tests passing in22.02s; model loading plus generation/plotting
took31.8s. All source/identity/morphology/measurement checks passed, with no
selected-object match dropped. Source draws/report live at
`doc/figures/v36_truth_corners_16622674/`. One of262144 draws for row239507
failed the pre-transport open-disk check; all other draws were valid, and no
post-transport failures occurred. No radius/flux invalidity occurred here.
The four observed marginal-CDF ranges are35.5–88.2%,34.5–45.6%,22.5–58.0%,
and29.0–65.6%. The plots do not show conspicuous marginal outliers at actual
truth, but do not establish joint calibration, population coverage or shear
derivative accuracy. See [matched-truth results](V36_INFERENCE_REVIEW.md#matched-truth-conditional-corner-plots--2026-09-21).

Visual inspection found legend/x-label crowding. Added a hash-verified
`--render-from` mode and adjusted only figure spacing; CPU job16622696 completed
in29s, re-rendering the exact saved samples into
`doc/figures/v36_truth_corners_20260921_final/`. Final visual inspection passed.
No production code or model was changed; the earlier mixture job stays paused.

## 2026-09-21 — Marginal corner plots for fainter problematic observations

Owner requested triangular contours with diagonal marginal histograms for
fainter failures. Added `scripts/plot_disk_faint_corners.py`, the corresponding
one-GPU launcher, and `tests/test_disk_faint_corners.py`. Chosen previously
audited rows412689/120835/86325 have MAG_AUTO19.4465/20.1248/20.5881: the worst
production-curvature row in19–19.5 and20–20.3, plus the faintest of the exact
worst32. This is a selected diagnostic panel, not a representative holdout.

For each row, retain the union of the nine-node top32 atoms from completed
exact24m references. Draw8192 physical samples per atom at the current center
and center+(.001,0), with atom-identity-seeded common random numbers. Recompute
output-conditioned disk transport for every sample. Use original equal-prior
times cached usability weights, apply the unchanged measured radius>0.6arcsec
and MAG_AUTO<25.8 cuts, then normalize within the retained atom subset. No
posterior atom reweighting, KDE, smoothing, model tuning or new cuts. This
produces genuine finite-sample1D/2D marginals, unlike the earlier density slices,
but NOT the full24m population predictive law. The top32 likelihood mass at the
observation is reported separately and is not integrated predictive coverage.

The Matplotlib skill informed the corner layout; the resource-check skill
informed oneA40/fourCPU/64GiB allocation, job16622565, with an empty user queue
before submission. Three focused tests cover mixture-selection weighting,
contour mass thresholds and plotting units; the two previous plotting tests
also run before generation. Ruff, Python compilation, Bash syntax and
`git diff --check` passed. The intended output directory was
`doc/figures/v36_faint_corners_16622565/`; no completed plot products were produced.
Job16622565 passed all five tests in24.23s, then failed on a forward draw with
non-finite or non-positive radius/flux before response evaluation. No corner
plots were completed. Invalid draws were not clipped or repaired. Their
frequency and originating atom have not yet been measured. The owner then
clarified whether these predictions were conditional on each observation's
actual truth properties. They were not: both this attempted follow-up and
the earlier slice plots use data-chosen prior-atom mixtures. This mixture work
was paused and superseded by the completed
[own-truth diagnostic](#2026-09-21--own-truth-conditional-corners-across-mag_auto1721),
using actual simulation truth and neighbour context, not those prior atoms.
No core implementation, pinned artifact, inference cache or production job
changed. The100k launch remains withheld.

## 2026-09-21 — Plot joint-density slices for problematic observations

Owner requested joint likelihood plots with observed properties overlaid.
Added `scripts/plot_disk_joint_likelihood.py`,
`jobs/job_plot_disk_joint_likelihood.sh`, and two focused tests in
`tests/test_disk_joint_likelihood_plot.py`. The diagnostic uses previously
inspected bright rows142230/3563 and their fixed node-wise top32 atom unions.
Six pairwise slices hold the two other outputs at the observation; these are
not pairwise marginals or credible regions. Original uniform-prior/usability
weights and full-prior cached selection normalization are retained. Disk
response is recomputed for each radius/flux output cell. Neural evaluation,
input shear and preprocessing are FP64; classifier probabilities are the
cached FP32 values. Contours compare the existing center and center+(.001,0).
No posterior reweighting, KDE, population cut change, or model tuning is used.

The Matplotlib skill informed the OO multi-panel layout and shared sequential
color scale. The resource skill informed a one-A40, four-CPU,64GiB Slurm job
16621509, within the two-GPU combined limit; queue was empty before submission.
Whole-node resource recommendations do not supersede the allocated resources.
Python compilation, Bash syntax and `git diff --check` passed before launch.
Initial job16621488 stopped in the new test on a2.8e-17 grid-center rounding
difference. Explicitly set each production grid center to the observed value
and use an appropriate floating tolerance for the synthetic linspace test.
Immediate retry16621500 apparently reused stale shared-filesystem bytecode
(traceback displayed the new allclose call but executed the old exact-array
assertion). A job-local bytecode cache in16621509 avoids that reuse; both
focused tests then passed in15.93s. Job16621509 completed in3m25s. Output goes to
`doc/figures/v36_joint_likelihood_16621509/`, including raw grids and provenance.
All six panel centers replay the direct observed-point density within1e-9.
Visual inspection found very narrow e1 peaks at the broad display scale and
an overlapping footer. Added a fixed local-window view and a rendering-only
mode; job16621540 re-evaluates finer local grids and redraws the saved broad
grids without changing any density. Its two focused tests passed in13.64s;
the full job completed in3m07s. The local half-widths are .015/.01 in shape,
.1arcsec in radius and15% of observed flux; these are display settings only.
Details: [joint density plots](V36_INFERENCE_REVIEW.md#joint-measured-property-density-plots--2026-09-21).
Matplotlib's outside figure legend still overlapped the explanatory footer;
changed it to an explicit figure-coordinate position. CPU-only job16621572
redraws both sets from their saved grids into
`doc/figures/v36_joint_likelihood_20260921/`, preserving the original arrays
and provenance; completed in32s. Reserved more bottom margin after inspection
to separate the legend from axis labels, with final CPU rendering job16621622
writing `doc/figures/v36_joint_likelihood_20260921_final/`. These layout-only
jobs do not load models or recompute densities. Final job16621622 completed
in32s; visual inspection confirms separated labels/legend/footer. Tests passed2/2; Ruff, Python
compilation, Bash syntax and whitespace checks pass. The plot lesson is a
narrow e1 ridge with the observation on its steep right flank in both examples;
radius/flux slices are multi-ridged. This is descriptive, not model rejection.
The diagnostic unions capture93.69%/71.69% at the center observation; captured
mass elsewhere on the plotted surfaces is unknown. No full-prior surface or
physical adequacy claim follows. No production code/cache or100k job changed.

## 2026-09-21 — Diagnose likelihood failure separately from sampling

Owner asked to look deeply into possible likelihood failure. Added diagnostic
scripts `probe_disk_likelihood_failure.py`,
`summarize_disk_likelihood_failure.py`, and
`probe_disk_derivative_reference.py`, their two GPU launchers, and
`tests/test_disk_likelihood_failure_probe.py`. No core `sbsi/` implementation,
model, production cache, population, or estimator was changed.

Protocol: two previously inspected failures (rows142230/3563) and original
rows0/1, not the reserved numerical holdouts. Sum all24m atoms at the existing
nine shear nodes, measure posterior effective atom count, and compare four
disjoint index-mod4 banks and their two12m unions. Bank comparisons concern
the numerator only: subset-specific selection normalization is not computed,
and these partitions are not independently generated priors. On each fixed
union of node-wise top32 atoms, recompute genuine classifier features and
probabilities at h=.001,.0005,.00025,.000125; compare compiled/eager density,
FP64 neural evaluation, and freezing each of probability/context/transport
at its center value. Report original-node captured mass. These truncated
numerators and deliberately frozen-component arms are not production models
or full-prior derivative validation.

A separate autodiff reference uses FP64 neural evaluation AND input shear/
preprocessing; the production helper retains FP32 raw-shape arithmetic even
when neural weights are promoted. The reference freezes the classifier and
compares finite differences down to h=.00003125 against autograd derivatives.
The reducer separates posterior-averaged component information from the
covariance of component scores, reporting the finite-step identity residual.

GPU array16621068 completed in11m21s/10m39s, using two A40s total and four
CPUs/64GiB per worker. Resource detection confirms one visible A40 per worker;
scheduler allocation, not whole-node recommendations, bounded the work.
One-GPU derivative job16621104 then completed in36s, with all five new tests
passing. CPU summary/full-suite job16621083 stalled in uninterruptible I/O
before Python started on cip-cl-compute3 (process inspection: shell child,
zero CPU, about1MiB RSS). Cancelled only that job, preserving logs. A first
replacement submission specifying inter,cip plus cip-cl-nv01 was rejected
without creating a job; CPU-only16621160 on partition cip/node cip-cl-nv01
completed in43s as the replacement. The full suite passed **302 tests,
2 skipped in27.37s**, followed by successful reduction. No100k production job
is submitted. Ruff, Python compilation, Bash syntax and `git diff --check`
pass. Reports are in the existing run's `likelihood_probe_16621068/summary.json`
and `derivative_reference_16621104.json`; all scientific GPU jobs are complete.

All-atom results: bright rows142230/3563 have6.29/16.58 effective posterior
atoms at the center, versus11636.98/292088.13 for rows0/1. The12m parity halves
give g1 numerator scores19.16 versus1314.33 and-5.96 versus1534.90. Existing
worst32 records have median88.88% single-atom mass at+g1, with32 different
dominant identities. This establishes serious atom-bank sensitivity in the
selected failures, not population-level prior convergence or model bias.

Full-FP64/autodiff on the dominant unions, with center-frozen classifier,
retains I11=-1.521m/-3.232m. Freezing flow context reduces I11 to2.60/-5.64;
freezing disk transport leaves the large effect. At h=.001, g1 scores
957.16/1520.06 differ substantially from autodiff390.91/766.12. Thus sampling,
prior resolution and finite-step errors are separate observed problems;
negative curvature on these unions is not merely FP32 cancellation. The
unions capture93.69%/71.69% of bright center mass (only3.27%/.57% for rows0/1),
so no full-prior fine-step or classifier-normalized derivative claim follows.
No normalizer was extrapolated to new nodes. Findings and table are in
[likelihood-focused diagnostics](V36_INFERENCE_REVIEW.md#likelihood-focused-diagnostics--2026-09-21).

The finite-step mixture decomposition at h=.000125 gives positive mean
component I11 (~61545/~117368) but much larger component-score variances
(~1.582m/~3.349m), accounting for the negative mixture I11. The reported
finite-step residuals are nonzero and preserved, not corrected away. This
supports rapidly changing atom responsibility as the dominant g1 mechanism
on these unions. It does not imply every individual component is concave;
some also have negative information in the other direction.

The scientific-critical-thinking skill motivated separating model adequacy,
finite-prior resolution, derivative numerics, and sampler error. The resource
skill motivated the bounded allocation and dependency. Existing moment
calibration and improved NLL are not full-density/shear-derivative validation.
Next: establish converged prior/derivative integration before attributing
failed shear recovery to physical model mismatch; separately validate bright
conditional density/response against images. No archived model training or
cancelled independent validation chain was restarted. This investigation did
not establish whether the flow is physically too sharp or correctly sharp
but inadequately integrated by this prior.

## 2026-09-21 — Review end-to-end inference and gate the requested100k run

Owner requested a full likelihood/prior/sampling review, caution about empirical
tuning, and two-GPU100k inference only when robust. Added
[review and launch gates](V36_INFERENCE_REVIEW.md#decision--2026-09-21), covering
disk change of variables, full-prior measured selection, uniform finite-prior
construction, exact-plus-complement sampling, finite-log bias, state-dependent
classifier geometry, derivative precision, solver convergence and uncertainty.
Linked it from `doc/INFERENCE.md`. **Production launch is withheld.**

The inverse Jacobian/auxiliary-marginal composition and stratified reduction
are consistent on inspection and in existing regression tests. Important
unresolved issues remain: original Gaussian retrieval has demonstrated support
failure; exact bright-object curvature can still be negative;24m prior
resolution and source-generator/runtime equivalence are not established;
selection64-QMC/h=.001 convergence has not been established. The17-input
classifier contains five hard annulus bins in addition to its four smooth
crowding features, so global smoothness must not be assumed. The normalizer
called `ExactPopulationNormalization` is atom-complete but still uses a
sampled flow integral. The standardized-output density omits a fixed physical
unit determinant that cancels in same-model shear derivatives, not in arbitrary
absolute density comparisons. No implementation/cached-science change was made.

Added `scripts/audit_disk_population.py`,
`tests/test_disk_population_audit.py`, and CPU launcher
`jobs/job_audit_disk_population.sh`. Job16617937 completed in2m14s after the
full suite passed **297 tests,2 skipped in41.31s**. Resource receipt shows no
GPU; allocation was two CPUs/12GiB, with BLAS/OMP limited to one thread.
The script checks frozen input/source hashes, rejects ambiguous or unmatched
truth joins, scans all20 prior shards and reserves rows without evaluating
new likelihoods. Output is under the existing run's
`population_review_16617937/report.json`.

Actual frozen500k support audit:54231 objects (10.8462%) have true-r>=26;
all source matches succeed. Prior:16387409/24000000 (68.2809%) atoms have
true-r>=26 and account for approximately47.14% of cached usability-weighted
mass **before measured cuts**. This is not selected/posterior mass or proof
of bias. It documents a material gap from the retained true-r<26 development
evidence; no independent model acceptance exists for this uncut population.
Prior cases20000–20199 and observed cases40–139 are disjoint.

Reserved a64-object numerical holdout (seed20260922),32 uniform plus8 per
fixed magnitude stratum, excluding first32 and all50 inspected worst rows.
It is held out only from proposal development, not independent image/model
validation. Separately reserved a uniform100000-row production sample
(seed20260923), with **no tail/development exclusions**, for two50000-row
workers after acceptance;10944 of these rows have true-r>=26. Array hash and
membership are persisted. No production job or automatic production dependency
was submitted. Subsequent proposal choices must be frozen before inspecting
holdout likelihoods; further tuning retires those holdouts as validation.

Current GPU array16617850 remains the only GPU allocation (two A40s total).
At review handoff each worker has completed three exact/comparison objects
and is progressing. It and reducer16617867 remain development diagnostics,
not launch gates by themselves. Scientific-critical-thinking guidance motivated
the development/acceptance separation and prospective numerical budget; the
resource skill confirmed allocation limits. No models, response scales,
empirical shear corrections, cuts, prior weights or production caches changed.

Validation: Bash syntax, Python compilation, system Ruff and `git diff --check`
pass. The py31 environment lacks the Ruff module; `/usr/bin/ruff` is used
instead. Next: finish the existing development-tail benchmark; separately
resolve the uncut support/prior, normalization, derivative and solver gates.
If resolution requires retraining, a changed population or restarting archived
independent validation, obtain an explicit owner decision. Task-wakeup tools
are unavailable; the existing Slurm reducer runs without waking the assistant.

## 2026-09-21 — Start exact tail validation with two persistent workers

The second all-atom benchmark16617754 completed in5m13s. Row3563 remains
non-concave under exact integration: information
`[[-2454379.98,267716.02],[267716.02,-251436.95]]`. Thus a successful proposal
repair must not be confused with proving positive full-catalogue curvature at
the current expansion point.

Student(df3) union probe16617759 completed in4m09s; Cauchy(df1) union16617785
completed in4m28s. Both preserve the original1024 exact atoms and the original
defensive complement sampler. Cauchy ranking moves the previously missed
33%/18%-mass atoms to ranks58/343. AtK65536/M16384 the worst object's g1 score
error falls from1116.814 to41.388 versus exact; the second object's error falls
from72.157 to9.829. This is a useful improvement, **not adequate evidence of
integration convergence**. No proposal variant is promoted to production.

Refactored the unchanged all-atom reduction into `exact_observation` in
`scripts/probe_disk_curvature.py` and added a blockwise-normalization regression.
Added `scripts/validate_disk_tail_integration.py`,
`jobs/job_validate_disk_tail.sh`, and `scripts/summarize_disk_tail_validation.py`.
The new panel covers the32 most negative audited observations, reuses the two
completed exact references, and compares Cauchy-union K65536/K262144 with16k
complement draws against full24m-atom sums. This is a selected failure-tail
benchmark, not an unbiased population subsample. The reducer reports errors
and an explicitly diagnostic partial replacement of32 contributions; it never
reports a shear estimate or covariance.

Array16617850 has exactly **two persistent one-A40 workers**,16 objects each,
with model/prior allocations retained and exact/comparison checkpoints saved
after each object. CPU reducer16617867 depends on array success. Allow roughly
one hour plus queue time, based on the measured exact-scan rate. The initial
startup16617841 was cancelled at56s before panel evaluation after identifying
the generic prefilter-size guard forK262144. The wrapper now declares its
original131072-location shortlist separately from the larger whole-prior union;
fresh output roots and freshly loaded code are used for16617850. No old output
was deleted or overwritten.

Both16617850 startup gates pass6 tests (10.99s/10.89s), including independent
SciPy checks of Student(df1/df3) scores, union uniqueness, and exact summation
across atom blocks. Resource receipts show two full46GB A40 allocations with
four CPUs/64GiB host memory each. Ruff, Bash syntax and `git diff --check` pass.
No other SBSI GPU work is active. Both workers have completed their first
comparison and exceeded one million atoms on their next exact observation.
AtK262144 the two existing benchmarks' g1 score errors are6.1435/3.6575,
versus1116.8140/72.1568 in production. The corresponding first-object
information errors are `[[-9803.50,428.99],[428.99,-7548.77]]`; this is a
substantial reduction but does not establish catalogue-level numerical tolerance.
Task-wakeup/automation tools remain unavailable; the Slurm reducer will run
automatically but does not wake the assistant.

Next: use the exact panel to decide whether the integration proposal is accurate
enough for a targeted recompute, and separately determine whether a safeguarded
recentered solve is required. Models, cuts, prior masses, normalizer and original
production moments are unchanged; the final positivity requirement stays intact.

## 2026-09-21 — Locate proxy support failure and test auxiliary retrieval

Whole-proxy probe16617724 completed in4m19s wall time. Increasing the same
Gaussian-proxy exact stratum toK131072 still substantially disagrees with the
all-atom result for row142230. The centre's two leading atoms carry32.90% and
18.12% of exact numerator mass but have proxy ranks961486 and1648131. Neither
is shortlisted; each has proposal probability about4.17e-9, hence only6.83e-5
expected occurrences in16k draws. This explains why even64k nested draws and
two seeds can look stable while missing most of the centre's mass.

Added `scripts/probe_disk_auxiliary.py`, its scheduled launcher and
`tests/test_disk_auxiliary_probe.py`. The diagnostic unions the unchanged
original1024 candidates with radius/flux-only Gaussian-proxy candidates, using
total exact budgets4096/16384/65536. Complement sampling and weights are
unchanged; union membership is unique and preserves every original candidate.
This is a proposed integration remedy, not a new likelihood or population cut.
Job16617745 completed in4m06s wall time; its startup gate passed both union tests
(2 passed in9.23s). Radius/flux-only Gaussian ranking also fails to recover the
dominant atoms (ranks850217 and1651496), so that proposed remedy is rejected.
Added an optional four-output Student-t(df3) ranking to the diagnostic union,
leaving the actual flow density and complement proposal unchanged. Its score
ranking is checked against SciPy's independent Student density. Dependent
job16617759 is running; startup gate3 passed in9.96s. The likelihood's physical
scales and models are not modified by this proposal-only tail choice.

The exact probe now accepts an explicit row from the audited worst-object list.
Job16617754 is checking row3563, the second-worst production row, against every
prior atom as a separate numerical benchmark. Together with16617759 this
preserves the two-GPU cap. Both proposals and benchmarks keep all observations
and the same finite prior; no full rerun has been launched.

Scheduled regression16617733:40 passed in22.50s across the existing sampler,
disk likelihood/cache/runtime-compatibility tests and new stencil regression.
Ruff, shell syntax and whitespace checks pass. The original500k moments remain
unchanged. A finite-prior/non-concavity problem also remains possible after
fixing integration; an exact benchmark, not positivity alone, gates the remedy.

## 2026-09-21 — Exact bright-object probe confirms curvature and integration error

Probe array 16617682 completed: exact arm 5m49s wall time, sampled arm 3m46s.
CPU summary 16617708 completed in 21 seconds. Added the hash-checked reducer
`scripts/summarize_disk_curvature_probe.py`; its report is
`curvature_probe_summary_v1.json` under the existing V3.6 run root.

For the worst production observation (row142230, measured MAG_AUTO17.3015),
exact summation over all24million atoms gives score `(926.5168,-472.9388)` and
information `[[-1686130.55,500735.41],[500735.41,-879639.81]]`, versus production
score `(2043.3309,-1210.4676)` and information
`[[-2707895.46,452212.67],[452212.67,-2134515.99]]`. Thus **both** genuine
non-concavity at the initial centre and material finite-draw integration error
are present for this object. The exact numerator's leading atom changes across
the stencil; atom9262565 goes from8.57% posterior mass at the centre to84.59%
at the positive-g1 node and97.44% at the positive-g1/negative-g2 corner.

The32 worst objects retain a large negative curvature contribution at16k,
32k and64k draws under both seeds8701/8702. More draws alone do not resolve the
problem. This targeted selection is diagnostic, not a population estimator.
No objects were removed and no correction/offset was fitted to injected shear.

Added `scripts/probe_disk_proposal.py` and `jobs/job_probe_disk_proposal.sh`.
Single-GPU job16617724 checks each dominant exact atom's original shortlist
membership, full-proxy rank and expected proposal hits, then compares whole-prior
Gaussian-proxy exact strata K1024/K16384/K131072 with16k complement draws on
the same64 diagnostic objects. It changes numerical integration only; models,
population, centre and full-prior normalization remain fixed. Ruff and shell
syntax pass. Startup and results pending. No production inference code or
results were overwritten, and no remedy has yet been accepted. Updated stale
introductory text in `doc/INFERENCE.md` and the runtime boundary in
`doc/JOINT_CALIBRATION.md` to distinguish the dedicated disk driver from the
unchanged default/legacy path and the stopped model-development investigation.

## 2026-09-21 — Diagnose failed V3.6 production curvature

All 20 production partitions of array 16606146 completed, covering all 500,000
observations. Combiner 16606157 failed on non-positive information at the first
draw rung. A CPU audit also finds non-positive information at every subsequent
rung, including the final 16,384-draw result: eigenvalues
`[-3687071.14567819, 71478092.02541134]`. No valid final estimate or covariance
has been published; the positivity guard remains intact.

Added `scripts/audit_disk_curvature.py`, checking partition/input hashes,
identities, exact observation coverage and moment finiteness, then separating
normalization curvature and tabulating diagnostic contributions by measured
properties and source case. Job 16617664 completed in 24 seconds; report is
`curvature_audit_v1.json` under the existing 20260920 run root. The negative
direction is dominated by bright observations, not normalization: numerator-only
information eigenvalues are `[-3874527.63, 70520765.56]`. The ten most negative
observations contribute about -19.51 million along that direction. Exclusion
tables are diagnostics, **not alternative estimates or authorized cuts**.

Corrected the interpretation of saved weight arrays: they have draw-rung and
observation axes. The final-rung median ESS is 893.97, not the 161.41 obtained
by pooling all rungs; 64 final-rung Pareto-k diagnostics are nonfinite, while
all final score/information values are finite.

Added `scripts/probe_disk_curvature.py`, `jobs/job_probe_disk_curvature.sh`, and
`tests/test_disk_curvature_probe.py`. The probe revalidates production cache,
model, input and response hashes without changing inference implementation or
old receipts. One arm sums all 24 million atoms exactly for the worst object;
the other uses paired 16k/32k/64k draws and two seeds for the worst 32 objects
plus the first 32 comparison rows. Array 16617682 has two one-A40 tasks, the
only active SBSI GPU jobs, each with four CPUs/64 GiB host memory. Resource
detection confirms full 46 GB A40s. Test job 16617683 passed the stencil-algebra
regression (1 passed in 12.15 seconds); ruff, launcher shell syntax and
`git diff --check` pass. Both probes have loaded the hash-validated runtime
in 81 seconds and are compiling/evaluating. Probe results pending. Updated
the inference and catalogue-prior status sections; old outputs remain intact.

Next: distinguish finite-draw integration failure from genuine non-concavity
of the fixed finite-prior likelihood before choosing a correction. No model,
population, normalizer, production moments, or positivity requirement changed.

## 2026-09-20 — Audit full-prior pilot and eliminate response-cache thrashing

All20 preparation shards16605612 completed in7m41s–8m19s each;
assembly16605621 completed in2m08s. Full-prior pilot16605622 completed:
24million atoms,256observations,16k draws, compiled density and16-object
chunks. All moment and diagnostic arrays are finite, all objects used16384
draws, and the summed information eigenvalues are59224.20 and119798.25.
Peak PyTorch allocation19,025,835,520B; initial load82.7s, inference257.04s.
Warm throughput201.43s/240objects projects29.14h at500k/four GPUs, excluding
startup. This exposed an implementation bottleneck, not a calibration failure.

`DiskCatalogueLikelihood._coefficients` had one cache slot, while the shared
stratified estimator alternates exact-candidate and complement grids at every
shear node. They evicted one another, recomputing identical response sums nine
times. Replaced it with a bounded two-entry LRU: no density, coefficients,
draws, probabilities, stencil or numerical precision changed. Added an
alternating-grid regression test, including bounded eviction and equality to
fresh evaluations.

`scripts/run_disk_inference.py` now permits this one audited runtime-only
change over the existing preparation receipts. It requires exact old/new
density hashes, the original producer-driver hash, identical source text for
all four preparation functions, every other implementation hash unchanged,
and every scientific identity unchanged. Receipts and arrays are not rewritten;
both producer identity and current runtime identity are recorded in results.
Tests reject other implementation/model changes. This is not a general
"ignore provenance" flag.

Validation16606079: **290 passed,1 skipped in40.45s**. The same256-observation
full-prior pilot is replayed, and `scripts/check_disk_pilot_replay.py` requires
bitwise equality of every saved array before writing its validation receipt.
Resource checks again verified a full46GB A40; the conservative96GB host
allocation and16-object chunks are retained. Ruff, shell syntax and
`git diff --check` pass. Replay16606079 completed successfully: every array
(moments, all ladder levels, draw/unique counts and weight diagnostics) is
**bitwise identical** to pilot16605622. Total inference decreased257.04s to
92.69s. Excluding each first16-object compile/warmup batch, throughput
improves201.43s/240objects to51.66s/240objects, a3.90x speedup. Peak GPU
allocation remains19.03GB. Four-GPU projection is7.47hours of warm inference;
allow8–10hours plus queue delays, subject to production throughput variation.

Added production launcher for20disjoint25000-object partitions with a combined
four-GPU throttle and a dependent strict full-coverage combiner. The launcher
requires the replay receipt and refuses implementation changes since that
validated pilot. No production job was submitted before replay validation.

**Production array16606146 is submitted**:20x25000observations, capped at four
full A40s,96GB host memory and four CPUs per worker, writing
`production_lru_v1/part_00` through `part_19` beneath the existing20260920 run
root. Dependent strict combiner**16606157** runs after array success; it refuses
missing/overlapping coverage, incompatible identities, hash mismatches and
non-positive combined information. Scientific settings remain uncut24million
atoms, measured radius>0.6arcsec/MAG_AUTO<25.8, and the full512–16384 ladder.

Startup verified for all four production workers on full46GB A40s. Partitions
0–3 passed the production gate, completed cache loading/compilation and
reported16or more evaluated objects; the latest check had304,272,256,16rows
respectively. Warm first-worker rates span roughly3.8–4.7objects/s, consistent
with the8–10hour planning range. No early error was observed. The combiner is
pending array success. Scheduled task-wakeup tools remain unavailable, so no
automatic assistant follow-up is installed; the scheduler chain continues.

## 2026-09-20 — Integrate disk inference and start full-prior cache preparation

### Driver and validation

Added `scripts/run_disk_inference.py` with prepare/assemble/run stages and
`sbsi/disk_inference_store.py` for validated subset loading, frozen shear-state
classifier probabilities, global proposal summaries and memory-mapped sharded
disk responses. It calls the existing `run_adaptive_section5` estimator;
no alternate estimator or additive-response approximation is introduced.
The legacy driver still rejects the preparation-only disk config. Default
likelihood and numerical configurations remain unchanged.

The new path verifies model, subset, feature-order, pairing, input and code
hashes. Assembly requires every source shard, rejects partial/mismatched
receipts, applies proposal dispersion floors globally, and combines full-prior
selected masses. Cached probabilities cover zero plus the nine stencil nodes;
missing nodes fail closed. Inference writes additive per-object moments and
all six512–16384 draw levels for the existing strict partition combiner.
The requested500k input fixes a common starting center
`(0.013912273363531775, 0.0002826679500570267)` without truth information.

Selection uses64QMC draws per atom, proposal summaries128, step0.001,
K1024/prefilter131072, and unmodified float64 tilted-score arithmetic.
Preparation streams use `SeedSequence([8101, source_shard])` for selection and
`SeedSequence([8201, source_shard])` for coordinates, with selection reset at
every shear point for CRN. This explicit sharded stream is not bitwise replay
of the historical single global stream. The subset and scientific cuts are
unchanged; no truth cut or empirical correction is added.

Full-A40 validation16605587: **288 passed,1 skipped in24.97s**. First attempt
16605578 stopped on an overly strict exact-equality test: NumPy target-column
indexing changes float64 reduction order by1.11e-16. The test now allows
roundoff, with no change to computed summaries. Ruff, launcher syntax and
`git diff --check` pass. Resource checks confirm full46GB A40s; workers honor
their four-CPU/48GB allocations, not the entire node's detected resources.

The real-model end-to-end smoke pilot completed on16384 prior atoms and128
real observations in23.03s of inference, peak PyTorch allocation154,863,104B.
It prepared ten classifier/selection states, reloaded hashed caches and wrote
finite moments at every draw level. This is an integration check, not a shear
calibration result or a full-prior runtime estimate. Artifacts live at
`inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1/pilot_16605587/`.

### Scheduled full-prior stages

The24million subset job16603639 completed successfully in20m08s: all20shards,
24,000,000 atoms and392,263,736 retained directed pairs. The completion
manifest and per-shard receipts were checked after the owner's notification.

New jobs:

- **16605612**, `jobs/job_prepare_disk_inference.sh`:20prior shards, at most
  four full A40 GPUs total, writing `disk_cache_v1/`.
- **16605621**, `jobs/job_assemble_disk_inference.sh`:CPU assembly after all
  preparation tasks succeed, writing `disk_assembled_v1/`.
- **16605622**, `jobs/job_disk_fullprior_pilot.sh`:256observations against
  all24million atoms,16k draws,16-object chunks and compiled density, after
  successful assembly. No500k observation inference array is submitted yet.

Startup was checked on all four workers: each passed source verification and
reached at least524,288/approximately1.2million proposal-coordinate rows;
shard0 reached786,432 rows at the handoff check. All four allocations are full
A40s, split across two compute nodes. No early failure was observed.

Check full-prior memory, finite moments, compilation behavior and timing before
launching production observation partitions. Any failed predecessor blocks
its dependent stage. Automatic task-wakeup tools are unavailable in this
session; these scheduler dependencies do not themselves monitor/report results.

## 2026-09-20 — Adopt an uncut uniform 24-million-atom prior

Owner selected24million atoms instead of the full139,936,000-row store.
`scripts/subsample_disk_prior.py` samples globally without replacement,
seed20260920, assigns equal1/24million masses, records original draw ranks and
source identities, verifies source hashes and preserves every selected
primary's response neighbours (including unsampled secondary atoms).
No truth-property selection is introduced. `DiskCatalogueModelCache` accepts
explicit original identities to keep classifier pair geometry correct after
compaction. Added subsampling/identity tests and the CPU preparation launcher;
updated the16k config description and
[prior documentation](CATALOGUE_PRIOR.md#v36-like-uncut-24-million-atom-subset).
Default models, numerical defaults and the preparation-only guard are unchanged.

The earlier2–3day estimate below is superseded: it combined the enlarged prior
with a virtual24GB A40 throughput probe. A matched full-A40 probe16603474 gives
19–23hours for tilted sampling alone on140million atoms; this is not an
end-to-end ETA. New24million-atom full-A40 probe16603624 tests16/32-object
batches. Its synthetic timings must not be reported as complete inference time.
All20 full uncut source shards have completed; source preparation is reusable.

Full-A40 probe16603624 completed: warm medians0.5468s/16objects and
0.6687s/32objects, projecting1.187h and0.726h respectively for tilted proposal
sampling at500k observations/four GPUs. Peak tensor allocations are11.72GB
and22.09GB, without resident likelihood tensors. This removes the oversized
prior bottleneck; complete runtime still requires the real-model pilot.
The resource skill verified a full46GB A40 and sufficient CPU-node memory;
preparation remains sharded/memory-mapped within its48GB Slurm allocation,
not the entire node's detected memory or CPU count.

Five targeted subsampling/cache tests pass in15.04s; Ruff, shell syntax and
`git diff --check` pass. CPU job16603639 completed the full suite:
**283 passed,2 skipped in52.08s**, no regression/snapshot failures, then began
materializing `prior_subset24m`. Its startup and resource report were checked.
First shard completed with1,200,254 sampled atoms and19,625,085 retained pairs
in53.7s including source verification; the second shard is in progress at
handoff. Top-level completion manifest is written only after all20 shards.
Production integration, normalization/proposal caches and a real-model pilot
remain next steps; no full V3.6 inference has been launched.

## 2026-09-20 — Begin uncut V3.6-like 500k/16k inference preparation

### Current status and runtime decision

The500k input is complete:18,157,963 eligible measurements in cases40–139,
no crossmatch-without-shape or matched-without-truth rows. Of the eligible
population,1,977,959 (10.89%) have true r>=26 and lie beyond the V3.6 training
parent. This percentage describes the eligible population, not an exact count
in the frozen500k subsample. No selected zero shapes were found. First full
feature shard completed with6,996,800 atoms and114,354,404 directed pairs;
feature construction took219.7s before receipt hashing. Remaining CPU shards
are array16603279,1–19 throttled to four tasks, dependent on first-shard and
test success. Startup was checked through source verification and pair progress.

Density/normalizer benchmark16603280 projects4.23h for density/response work
plus3.45h for128-draw proposal coordinates and64-draw nine-node normalization
on140million atoms, on four GPUs. Crucially, full-size synthetic proposal
benchmark16603325 measured2.5847s per two observations at139,936,000 atoms:
**44.9h for tilted sampling alone** at500k observations/four A40-24Q GPUs.
An opt-in blocked float64 cumulative sum (`WholeCatalogueProxy.cdf_block_size`)
reduces the same probe to2.1235s/two observations in16603358, about36.9h.
It changes addition order, not proposal probabilities; four boundary/ragged
tests pass in16603359. Historical sampling uses the original scan by default.
The production32-object proposal batch would exceed memory on this enlarged
prior; a memory-bounded batch is required before launch. Synthetic benchmarks
exclude resident likelihood tensors and local candidate retrieval.

Thus the revised planning estimate is **2–3 days on four A40s**, not the
earlier inference-kernel-only4.4h estimate. No multi-day GPU inference or
normalization array has been submitted. Await the owner's direction on this
resource commitment; CPU truth-feature preparation may finish independently.
There is no available automation/scheduled-wakeup tool in this session, so no
automatic follow-up has been created.

### Implemented components and remaining integration

`DiskCatalogueLikelihood` now routes exact, sampled, ragged tensor and
conditional-density paths through the inverse Möbius map and its Jacobian,
with observed radius/flux-conditioned coefficients reused across stencil
nodes. `DiskCatalogueModelCache` rebuilds all17 classifier features in chunks
for every shear state, retaining probabilities instead of huge feature frames.
Tests compare direct inverse-density calculations, state-dependent mass,
padding/invalid outputs, coefficient reuse, and the original classifier feature
constructor. Full suite16603250: **275 passed,2 skipped in43.29s**; targeted
density16603200:4 passed. `configs/inference_v1_2_16k.json` records the extended
draw ladder without changing defaults; its prior remains explicitly unbound.
The disk components are not yet wired into persisted model-cache loading,
distributed preparation, the production driver or full-scale memory management.
The preparation-only guard remains essential: do not remove it until those
paths and a real-model end-to-end pilot pass. No inference result exists yet.
Final CPU regression job16603398 reports **279 passed,2 skipped in37.26s**,
with zero regression/snapshot failures. Ruff passes for all files changed by
this preparation turn; `git diff --check` and launcher syntax checks pass.
At handoff8/20 truth-feature shards are complete (55,974,400 source rows),
four CPU shards are progressing, and no GPU job is active or pending.

### Initial preparation implementation and validation

Owner requested the previous full inference with V3.6-like, no truth-property
cuts, measured radius strictly above0.6arcsec, twice the complement draws,
and up to four GPUs. This is newly authorized inference, not a restart of the
cancelled independent image-validation campaign. The four-GPU allowance applies
to this run. Current preparation uses CPUs; the bounded benchmark used one GPU.

Added preparation-only `configs/likelihood_v3_6_like.json`, pinning the three
models and strict radius/flux predicates. `run_inference.py` explicitly refuses
this config until the catalogue density/state-major classifier adapter is
integrated, preventing an accidental additive-response run. Default configs
are unchanged. Extended `OutputCut` with explicit `NAME:>LOWER:UPPER` bounds;
old half-open bounds retain their behavior and cache identity. Extended the
ConstGold adapter with explicit no-truth-cut flags and the V3.6 usability
definition, including valid zero shapes. Reports count selected true-r>=26
objects separately: the requested uncut population extrapolates beyond the
model's true-r<26 training parent. No offset or calibration correction is added.

Added `CatalogueDiskResponse`, an exact CSR pair-sum evaluator on arbitrary
atom/output triples. It pools only identical tree split cells and handles
isolated atoms as empty sums. Added an uncut, neighbour-complete truth-feature
preparer: original zero contexts are hash/identity checked; nearest20 neighbours
within10arcsec use the pinned V3.6 development pairing convention, without
truth-property cuts. This store is not yet a normalized production prior.

Validation: response pooling tests pass; Slurm16603127 reports **270 passed,
2 skipped in48.15s** for the complete active test suite. Benchmark16603029
failed on a missing PYTHONPATH;16603043 started before its parent-key correction
and failed on the old `ids` field. Both setup issues were corrected.
Successful A40-24Q benchmark16603054 verified the pinned model and source hashes:
262144 atom/output terms,3916572 pair/tree evaluations in1.06254s; nine warm
density calls in0.97518s. Linear projection is **4.422 GPU-parallel wall hours
on four GPUs for500k objects and16384 atoms**, excluding preparation, classifier
states, normalization, retrieval, I/O, queue time, and first compilation.
This is a one-scene uniform-atom eager-kernel benchmark, not a production timing
or inference result. Receipt: `logs/v36_benchmark_16603054.json`.

Preparation job16603127 first freezes500k uniformly sampled plus-leg objects
from cases40–139, seed20260914, then builds the first uncut prior-feature shard.
Run root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1`.
Startup was checked through passing tests and successful catalogue processing.
Production inference has **not** been submitted. Next, after the runtime
decision above: finish production integration, build all new
proposal/normalization caches and launch at most four GPU workers. Retain
K1024, h0.001, the one-step estimator, full-prior64-draw CRN normalization and
the paired8192/16384 draw ladder. Do not reuse V3.5 probabilities or cut caches.

## 2026-09-20 — Stop investigation, retain V3.6-like, and archive research code

Owner requested cancellation, the name V3.6-like for the retained single-flow/
disk-emulator/smooth-classifier set, and repository cleanup. Cancelled the
downstream jobs first (16595003,16602287,16602111,16595004,16595000,16594999),
then generation16593294 at about11:10 CEST. Accounting records 34 completed
scenes and six cancelled generation tasks; all pending stages are cancelled.
Queue inspection confirms no remaining research jobs. External artifacts,
partial products, original protocols, and scientific receipts remain intact.
The 40-scene validation produced no bias result and is closed by owner request.

Added `configs/models_v3_6_like.json` with the original three model identities,
four artifact hashes (including emulator metadata), source-manifest hash,
population/cuts, retained endpoints, and explicit cancellation status.
`sbsi.models` and CLI expose V3.6-like alongside the historical V3.5-like preset.
The new disk loader checks model/metadata hashes; the additive loader rejects
disk models. Extracted five unchanged feature functions into `sbsi.crowding`
and added a seventeen-input constructor. Updated model/API/README/environment/
convention/inference documentation and AGENTS.md. Inference defaults and model
weights are unchanged; V3.6-like is not a drop-in additive likelihood config.

Archived 453 retired files (150 scripts, 199 jobs, 90 tests, 7 plots, 4 experimental
package modules, 3 detailed documents) and 19 before-cleanup snapshots under
`archive/research-2026-09-20/`. Every archived file has an original path, archive
path, SHA-256 and byte count in `manifest.json`. All 472 hashes verified after
the moves. This preserves pre-existing uncommitted work and all original
scientific source bytes. The full previous work log and joint investigation
are archived; current docs summarize the retained model and link to history.
See [repository organization](REPOSITORY.md#research-archive).

Validation: active Python AST syntax and four launcher shell checks pass;
Ruff passes for the modified/new Python files. AST comparison confirms the
five extracted feature functions are unchanged. CPU verification 16602609
completes in 1m20s with empty stderr: `python -m pytest tests -q` reports
**267 passed, 1 skipped in 56.39s**, zero regression/snapshot failures.
`python -m sbsi validate-model V3.6-like` verifies all four artifact hashes.
The actual flow, classifier and disk emulator load; feature-order/training
metadata checks and a small flow/classifier prediction smoke pass. Active
imports have no missing or archived dependencies; all active document links
and their section anchors resolve. Final active counts are 31 package modules,
19 scripts, 4 launchers, and 35 test/support files. The verification used no GPU.
Final scheduler inspection is empty; `git diff --check` also passes.

Limitations: existing endpoints remain development evidence; fresh multi-axis
validation was stopped. Archived launchers need deliberate restoration in an
isolated checkout and source/hash checks before reuse. No science work or
automatic monitoring is scheduled by this cleanup.

## Earlier history

- [Complete log before 2026-09-20 cleanup](../archive/research-2026-09-20/doc/WORKLOG.md)
- [Joint calibration experiments](../archive/research-2026-09-20/doc/JOINT_CALIBRATION.md#goal-and-retained-population)
- [Earlier research archive](REPOSITORY.md#research-archive)

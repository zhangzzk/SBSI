# Gold-V2 — Staged forward-model shear calibration (PLAN)

**Status:** PLAN, locked + iterated with owner 2026-07-23. Parallels `Gold-V1.md` (the certified V1
shape result). **Supersedes** the additive framing `m = R_sim/(R_flow+R_blend+R_sel+R_detect)−1` and
the deleted Gold-v2 selection-*estimator* (cont.119). There are **no empirically-learned R_sel /
R_detect terms.** Stage 1 was resolved on 2026-07-24 (cont.146: the tabular S2 reframe, DeepSets
dropped); the V3 milestone flows (`MILESTONE.md`) descend from it. Implementation paths below
predate the library restructure and now live under `archive/pre-v3/`.

---

## 0. One line

Train **one** forward model `P(measured | truth, blending)` (measured = shape, flux, size — as
OUTPUTS; conditioned on TRUTH). Two independent **directions** can turn it into a shear estimate:
**(A) forward-modeling response** and **(B) Bayesian inference**. **We do (A) first.** In (A) the shear
response is computed by perturbing the **full truth** — not only ellipticity but the properties
(size/moments, flux) too — through the model. Built and validated in three stages, shape → selection →
detection, each gated by its **triad test**.

---

## 1. The model

- **No model conditions on g.** Flow and detection both condition on truth (+ blending) only; the SOLE
  source of g-dependence anywhere is perturbing the truth (§2). (Verified in `train_joint_forward.py`:
  the ±δ response contexts shear the *intrinsic ellipticity*, not a g input.)
- Flow: `P(measured | truth, blending)`. Conditions on the **true** primary properties (ellipticity,
  size / second moments, flux, sersic, …) + the neighbour / blending set. The **measured** quantities
  are OUTPUTS — the reframe that removes the errors-in-variables floor. Measured
  vector = **(e1, e2, flux, size)** — shape plus the two extra directions (flux, size).
- Detection: `P(detected | truth, blending)` — **also not conditioned on g, same as the flow.** Its
  shear response comes only from perturbing the truth (§2). Two ways to model it — (1) a detection head
  inside the flow, or (2) a **separate classifier**. **We do (2) first.**
- **FIREWALL:** the model trains on the half-shear / `det_meas` legs only. **constgold is EVAL-ONLY**
  — the held-out acceptance metric, never trained on (`CONVENTIONS.md` §2a).

---

## 2. Two directions — we pursue FORWARD-MODELING RESPONSE first

The one forward model can be turned into a shear estimate two ways. They are **independent**; we do the
first now and keep the second in view so they are not conflated.

### Direction A — Forward-modeling response  ◀ CURRENT FOCUS
Compute the shear response of the estimator **analytically**, by perturbing the **truth** through the
model, then form `m = R_sim / R_model − 1`.

The key point: **the shear perturbation acts on the full truth, not only the ellipticity.** A shear
maps the surface brightness `I(x) → I(A⁻¹x)`, so the true second moments transform `Q → A Q Aᵀ`.
To O(g) this means:
- the true **ellipticity** picks up the shear (the usual shape response), and
- the true **size / second moments** shift: `T = Qxx+Qyy → T·(1 − 2 e·g)` — an O(g) change
  **proportional to the galaxy's own ellipticity** (the spin-2 size coupling; this is exactly the
  size-vs-shear slope the half-shear gate measured), while
- the true **flux** is ~conserved.

So the shear perturbation of the truth is a **vector over (ellipticity, size/moments, flux)**. Pushing
that full-truth perturbation through `P(measured | truth, blending)` gives the shear response of
**every** measured quantity at once:
- measured-**shape** response → the **shape** bias (Stage 1),
- measured-**size/flux** response → the **selection** response once a cut is applied in measured space
  (Stage 2), since the cut boundary sits in measured space and moves with shear.

All response channels are **derived** from the one model by perturbing truth — none is a separately-fit
R. The estimate uses the total `R_model` under the applied cut; the triad R1/R2/R3 (§5) just decompose
it for validation.

### Direction B — Bayesian inference  (LATER, not now)
Invert `measured → truth`, apply cuts in measured space, estimate g from the forward-model
likelihood/posterior with the cut baked into its normalization. Handles selection & detection by
construction. **Deferred** until Direction A is nailed; recorded here only to keep the two directions
distinct.

---

## 3. Staged build — one stage at a time, each gated by its triad leg

### Stage 1 — SHAPE  (☑ reproduced on V2, cont.121 — half-shear shape m = −0.52% / +0.09%
for the lam_theta=100 / 400 pins; R_model ≈ +0.71)
Gold-V1 certified the shape result (**m = +0.245%**, 16-seed ensemble) with a *measured*-conditioned
flow. V2's forward model is a **different** object (truth-conditioned, measured as outputs), so the
shape number must be **reproduced on V2**: perturb the true ellipticity through `P(measured|truth)` →
measured-shape response → `m` on constgold. Target = V1's number (~1% tolerable single-seed).
- **Triad R1 (SHAPE):** matched pairs + TRUE cut + MEASURED shape, per-object antithetic diff →
  R_self + R_blend. Gate: the V2 model's shape response matches R1 / reproduces V1's `m`.
- Blend handled either as a conditioning input to the model or via the BlendEMU term as in V1 — to be
  pinned down in this stage; the milestone is matching V1's shape `m` with the V2 model.


**⚠ V1→V2 ablation (2026-07-24) — the V2 model UNDER-responds on the isolated shape self-response by ~7 pt;
cause = DeepSets ARCHITECTURE, not the truth-conditioning.** A one-knob-at-a-time ablation from certified V1
toward this V2 reframe, every step scored on ONE ruler (`flow/R_hs−1` on the g0.05 half-shear ISO acceptance
set: both-detected matched, true-cut Re>0.3 & mag<26, nn_bright>7″; `scripts/eval_selfresp_gap.py` / `_v2.py`,
branch `ablation-v1-to-v2`, 3-seed):

| step | change from V1 | ISO OVERALL flow/R_hs−1 |
|---|---|---|
| S0 | V1 as-is (measured-cond, 2D, blinded mean head) | **+1.28%** |
| S1 | measured→**true** conditioning | +0.32% |
| S2 | 2D→**4D** output (measured mag/size as outputs) | +0.27% |
| S3a | +6×3×5→**6×9×5** target grid | +4.70% |
| S3b | −intrinsic-shape **blinding** | +1.68% |
| — | **V2 (DeepSets), count-weight** | **−2.69%** |
| — | **V2 (DeepSets), equal-weight** | **−5.24%** |

Decomposition of the ~10-pt V1→V2 gap: DeepSets **architecture ≈ 7.4 pt** (dominant; S3a +4.70% → V2-count
−2.69% at identical grid & weighting) + **equal-weighting ≈ 2.5 pt** (secondary) + conditioning / output-dim /
grid / blinding / detection / loss ≈ **0** (all excluded). **The truth-conditioning + measured-as-outputs
reframe (§1) is NOT what costs the shape response** — that was the concern (the errors-in-variables
motivation for the reframe); the ablation RETRACTS the "EiV floor" attribution.
The deficit is V2's `SetConditionedForwardModel` shared **trunk** smearing the mean-head response (the neighbour
set is EMPTY for isolated galaxies, so it's the primary-feature trunk path, not neighbour handling). V1 avoids
this precisely via §1a's design (blind the intrinsic shape from the density flow; explicit mean head carries the
response). **Implication for Stage 1:** matching V1's shape `m` on V2 needs an ARCHITECTURE change to the V2
mean-head/trunk (give it a clean V1-style response channel), plus prefer count-weight over equal-weight (free
~2.5 pt). **Fix attempt #1 refuted:** an additive shape-skip `Linear(e1/e2_input_p → mean-head)` made it WORSE
(−2.69% → −4.37%), so the fix is deeper in the pooled-set/trunk representation, not the shape input channel.
Also note V1's own tightness is a **population-average CANCELLATION** of ±25% per-bin errors ([0.30,0.38)=−22.9%,
[0.50,0.75)=+14.5%), same per-bin structure V2 has — relevant for any per-bin/tomographic use. Full record:
WORKLOG cont.135–144.

**✅ RESOLVED (cont.146) — the fix is to DROP DeepSets for the shape response, not patch it.**
The "V1 duplicated, changed ONLY to true-conditioning + 4D output" model (= ablation rung S2: tabular
`ConditionalMeanFlow`, V1's blinded mean-head channel intact) closes constgold cleanly. `validate_constant_with_blend.py`
runs UNCHANGED on it — it reads the checkpoint's declared TRUE conditioners (`r_input_p`, `Re_input_p`, …)
straight from constgold, so no `--meas-prim-lookup` is needed; emulator R_blend as in V1. 3 seeds s501/502/503
(swaavg): R_flow = 0.2891 / 0.2964 / 0.2940 → **⟨R_flow⟩ = 0.2932 (V1 certified 0.2930±0.0032 — identical)**;
per-seed m = +1.13% / −0.49% / +0.04% → **ENSEMBLE m = +0.21%** (R_sim=0.4534, R_blend=0.1593, both = V1).
So the shape response is FULLY recovered; the −5.24% was ENTIRELY the DeepSets trunk. DeepSets is only needed
for a single-model blend response by perturbing individual neighbours — but V1's two-piece structure gets R_blend
from a separate emulator, so it is unnecessary here. **Fix attempt #1 (shape-skip on DeepSets) is moot — superseded
by dropping DeepSets.** Ckpts `.../ablation/measurement_flow_g0_ngmix_ablate_s2_true4d_s50{1,2,3}_swaavg.pt`;
closure `jobs/job_s2_constgold_closure.sh` (job 15222198). 16-seed cert ensemble (s504–516) training in flight
(job 15222703). Caveat: S2 ckpts trained at `--max-rows 4000000` (small-GPU cap) yet reproduce V1's R_flow;
full-row retrain optional for a formal cert.


**➕ Flux/size shear-response extension (cont.147-149, 2026-07-24).** The 4D flow's measured mag/size OUTPUTS (dims 2,3) initially had the WRONG-SIGN shear response (size orientation-coupling b_size = −0.50 vs truth +0.566) because only shape (dims 0,1) was response-supervised — dims 2,3 were NLL-only. FIXED by adding a theta-coupling pin on dims 2,3 (`--coupling-weight`/`--coupling-target-npz` in the V1-format trainer; its own 6×9×5 binid, shape pin untouched). λ_θ=500: **b_size −0.50 → +0.563** (truth +0.566, −0.6%, 3-seed consistent); constgold shape m stays in spec (3-seed ensemble −0.25%, vs S2 +0.21%, within seed scatter and |m|<0.3%). So the model now carries the correct shear response on shape, size, AND flux — the measured-observable responses the SELECTION-bias term needs. Eval `scripts/eval_fluxsize_response.py` (ghat-leg SHAPE control built in). Full record: WORKLOG cont.147-149, memory project_fluxsize_response. Branch ablation-v1-to-v2.

### Stage 2 — SELECTION  (◐ R2 measured cont.150: mostly closed, size>4.4 gap)
**Status (cont.150):** `scripts/eval_selection_response.py` pushes the moving MEASURED cut through the
pinned 4D flow and measures R2 against the det_meas half-shear truth (ISOLATED, R_blend~0). Harness
validated: NO-CUT control m=-0.49%; TRUE-cut NULL boundary=+0.0000 exactly. Measured-cut suite (3-seed
lt500, N=1.02M): mag<24/24.5/25 m=+1.24/+0.43/-0.00%, size>2.9/3.5 m=-0.68/-1.14% -> selection CLOSED
to ~+-1% for mag + mild-size cuts (selection shifts the sim response by up to +18.6%, and the model
tracks it). ONE GAP: size>4.4 (43% kept) m=-6.6%, model over-predicts the deep-size-tail moving
boundary. DIAGNOSED (cont.151): shape-size COVARIANCE, not marginal -- fracM==fracS at every deep cut
(size>4.4 0.43=0.43, >5.0/5.5 too), so the size marginal + mean b_size are right; the model over-predicts
how strongly the large-size subsample's SHAPE responds to shear. FIX = Direction-A (perturb TRUE size ∝ e·g
so the size response AND its shape-correlation emerge self-consistently, replacing the mean-only pin).
PIN confirmed FREE: paired Delta_m(pinned-baseline)=-0.25%+/-0.47% over 6 seeds. Remaining: Direction-A;
compose selection with R_blend/constgold (currently ISOLATED-only).
**Status (cont.121):** the `--lam-theta` pin reproduces the HELD-OUT measured-size response
b_size(log) = +0.475 / +0.478 (target +0.490) and b_mag = +0.024 (target +0.025) with NO shape
trade-off (R_shape ≈ +0.71); the un-pinned flow was wrong-sign (b_size −0.077, b_mag −0.226).
Readout: `scripts/readout_rtheta.py`. REMAINING: apply the moving measured cut through the
model on constgold and show the selection response is closed (the NEXT-c overlay below).
Perturb the **full truth** (§2 Direction A: ellipticity AND size/moments/flux) through the model to get
the measured-size/flux shear response; apply the analysis cut in measured space; the moving cut
boundary produces the selection response — **derived from the model**, not fit.
⚠ The current trainer's ±δ contexts shear ONLY the ellipticity (not the true size), so the size
response does not emerge — which is exactly why `--lam-theta` (target
`results/response_target_theta_coupling_*.npz`) injects it by hand. The Direction-A-correct fix is to
**perturb the true size/moments too** (∝ e·g, §2) so the size response emerges from the model and the
pin becomes unnecessary. Decide this at Stage 2 — the pin is an interim workaround, valid only as
size-response supervision (analogue of V1's `lam_r`), validated against R2, not trusted blindly.
- **Triad R2 (SELECTION):** matched-detection + leg-average + MEASURED cut + INTRINSIC e →
  pure moving-boundary response; **must vanish for TRUE cuts**. Gate: `m` within target across the
  fixed measured-cut suite: **size 2.5/2.9/3.5/4.4, mag 24/24.5/25, S/N 10/15/20** (extreme cuts e.g.
  size>5.5 are diagnostic/resolution, not acceptance).

### Stage 3 — DETECTION
Model detection as a **separate classifier** `P(detected | truth, blending)` (Direction 2 — do this
first; Direction 1 = a detection head inside the flow is the alternative). **Not conditioned on g** —
its shear response comes from perturbing the truth (ellipticity + size/moments), exactly as for the
flow — so detection's g-dependence enters `R_model` as the detection channel.
- **Triad R3 (DETECTION):** leg-average + TRUE cut + INTRINSIC e, detection free to differ → pure
  detection response. Gate: R3 matches and `m` holds once detection-affected samples enter.
- `sbsi/detection_classifier.py` exists but was g-conditioned — the V2 classifier drops the g input.

---

## 4. Goal / acceptance

- **|m| < 0.3%** (ensemble over seeds). A single seed may be **~1%** (per-seed flow noise ≈ ±1%; the
  ensemble averages it down — V1: per-seed −0.85…+1.48%, 16-seed ensemble +0.245%).
- Evaluated on **constgold** (firewall: eval only): the true-cut deliverable, plus the fixed
  measured-cut suite for the selection stage.

---

## 5. Triad tests = validation harness (NOT the calibration)

`scripts/eval_joint_triad.py` measures R1/R2/R3 directly from the sims with an orthogonal design that
turns the other two channels off. These are **checks** that the forward model's responses are right at
each stage — the calibration itself is Direction A (§2).

---

## 6. Discipline (why this converges)

1. **Do the stages in order.** Do not start Stage N+1 until Stage N's triad gate passes.
2. **Direction A only for now.** No Bayesian-inference estimator until A is done.
3. **No empirical R_sel / R_detect additive terms** — every response channel comes from perturbing the
   one model's truth.
4. Gold-V1 is the certified **shape benchmark** to reproduce; do not re-litigate the shape physics —
   match V1's `m` on V2 and move on.

---

## 7. Artifacts / pointers

- Forward model: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_<tag>_seed<seed>_joint.pt` (+ `validation_<tag>_seed<seed>.{npz,txt}`).
- Trainer: `scripts/train_joint_forward.py` (`P(measured|truth)` flow, not g-conditioned; also carries
  a detection head = detection Direction 1; `--lam-theta` = interim size-response supervision).
- Size-response target: `scripts/build_theta_coupling_target.py` → `results/response_target_theta_coupling_c0-99_6x9x5.npz` (RAWfine 6×9×5 grid).
- Detection classifier (Direction 2, preferred): `sbsi/detection_classifier.py` +
  `scripts/train_detection_classifier.py` — **exists but was g-conditioned; the V2 version drops g.**
- Triad harness: `scripts/eval_joint_triad.py`; half-shear gate output `sbsi_caches/derisk/triad_halfshear_fixed.npz`.
- Benchmark: `Gold-V1.md`. Firewall/data: `CONVENTIONS.md` §2a.
- **Real next code:** (a, Direction A) response estimator that perturbs the **full truth** (ellipticity
  + size/moments + flux) through the V2 model, applies the measured cut, forms `m = R_sim/R_model − 1`;
  (b) a **not-g-conditioned detection classifier** (Stage 3, Direction 2). Direction B (invert → cut →
  estimate g) and detection-in-flow (Direction 1) are the *later* alternatives, not current targets.

---

## 8. Neighbour / blending scale convention — 7″ for shear response (owner 2026-07-23)

There are **three** catalogues, each carrying neighbour info defined at a **different angular
separation**: the **self-response (R_self)**, the **R_blend**, and the **detection** catalogue. The
pre-baked `neighbored` flag (and the `distance` column) in the response catalogues comes from the
**detection** convention — a **3″** search radius. **3″ is the wrong scale for anything shear-response
related.**

**Decision:** for shear-response work (isolation cuts AND R_blend prediction) use **7″** — the range the
**BlendEMU R_blend emulator was trained on** (10″ is a possible alternative; 7″ chosen for consistency
with the emulator). Concretely:

- **Isolation** (for the Stage-1 self-response test and the "isolated / R_blend=0" band): an object is
  isolated iff it has **no TRUE neighbour within 7″** — NOT the 3″ `neighbored` flag. Objects with a
  true neighbour at 3–7″ are **blended**, not isolated, and must receive an R_blend term.
- **R_blend prediction:** when the emulator predicts R_blend for a detection, feed it **all true
  neighbours within 7″** of that detection (not just a 3″ companion).

Consequence: the earlier "isolated" self-response gap (constgold +0.60 vs det_meas primary-only +0.49,
cont.125–126) was measured on the **3″-isolated** set, which wrongly includes 3–7″ blends. Re-measure on
the **7″-isolated** set; the gap is expected to shrink once 3–7″ neighbours are correctly treated as
blends handled by R_blend. This must be re-checked before attributing the residual to an undetected
diffuse coherent background.

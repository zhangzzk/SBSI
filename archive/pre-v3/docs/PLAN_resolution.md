# Plan: fix the resolution problem

**The issue.** Global constgold `m = -0.123 +- 0.152%` is excellent. Per-bin it is not: residuals span
`-4.12%` to `+5.65%` in true size, `m` reaches `+1.49%` (7.3 sigma) at `tRe > 0.6` and `+1.77%`
(9.0 sigma) on the joint true cut. The global number is right by cancellation. Per
`LITERATURE_resolution.md`, this is *grouping loss* / a *resolution deficit*, and both the weak-lensing
and statistics literatures say the global number is the wrong deliverable.

Drafted 2026-08-02. Phases 1-6 are still a **proposal**, not agreed. **Phase 0 has now been RUN** —
see the status block immediately below and WORKLOG `2026-08-02c`.

---

## STATUS after executing Phase 0 (2026-08-02)

| gate | outcome |
|---|---|
| **0a** additivity on isolated rows | **BLOCKED — not executable.** Only 3,534 of 11.67M rows (0.03%) have no neighbour flux within 7"; 1,855 also satisfy the emulator. That set carries **±11.1%** on its mean response, so it cannot adjudicate a 2-3 pt effect. The plan's premise ("for isolated galaxies `R_blend = 0` by construction") has no workable sample in constgold. Does NOT show additivity fails; does NOT clear the flow |
| **0b** extraction vs true size | **No size-dependence detected** (flatness χ²/dof 2.2/5), own-set and common-set agree so it is not selection. **A weak bound, not a clearance** — per-bin errors are not small against the 0.089 required-blend range |
| **0d** leg-matching per bin | **Real, significant, but does not fire the gate on the size axis.** Size: monotonic −0.84% → −1.72%, almost all of it a CONSTANT offset — bin-to-bin span 0.93 pt, ~9% of the residual's 10.6 pt span. **Blend axis: −0.15% → −3.12%, span 3.06 pt, larger than the 1.51 pt total residual there** — that one matters for Phase 4 |
| **0c** matched keep fraction | **DONE, and the caveat it closed was real.** `TRUE Re>0.6"` and `MEASURED>0.60"` had been read against each other while removing **36.7%** and **96.7%** of the population. At matched severity the two diverge monotonically — true cuts +1.43→+1.75 and nearly flat, measured cuts −1.17→−4.32 — **dm reaching −6.06 pt at 27% keep**. The masks are FROZEN, so no boundary term is involved: it is purely the response of the retained population. **Supports Phase 3b** |

**The load-bearing result is not any of those gates.** Phase 2's premise was tested directly on the
firewall-clean per-pair ruler and **does not survive**:

- Summed `R_blend` vs **true size**, emulator against ruler truth, at the fiducial tag: every bin
  within ~1.9σ (**overall +1.73% ± 3.25**; at a 3" aperture within ~1σ). Ruler truth shows no
  significant size trend either. The constgold inference implies a required variation of ≈**77%** of
  the emulator's mean prediction across size bins — excluded by the ruler at many σ.
- What the ruler DOES confirm is the **close-pair deficit: −40.9% below 1"** (−8.7% at 1-2", −2.1% at
  2-3"), reproducing the recorded −41.5% at the tuned tag.

**So the emulator's measurable defect is SEPARATION, not TRUE SIZE.** The size-axis residual that
2026-08-01s charged to the emulator is most likely the FLOW's own size error, re-labelled by the
`required_blend = r_sim − R_flow` construction which assigns 100% of the flow's error to the blend
term. **Phase 2 is struck as written** (see Phase 2 below). Phase 3a gains priority. Phase 1 is
unaffected and unblocked.

**Does the size budget then close on the flow? Only partly** (`scripts/eval_size_budget.py`). If the
emulator is right, the model residual IS the flow's error, which fig5 measures independently.
`corr(implied, fig5) = +0.546` — related, but not the same magnitude. Splitting offset from scatter:
the 0d leg-matching term **removes the offset almost exactly** (excluding the smallest-size bin, the
unexplained mean goes from **−1.55 pt to +0.02 pt**), leaving **1.64 pt of bin-to-bin scatter** plus
**one outlier at Re ≈ 0.35 of +7.7 pt** — which is the flow's TRAINING-DOMAIN EDGE (`Re > 0.3`).
A binning fix was made along the way (fig5's own edges were being paired by INDEX with constgold's)
and proved numerically immaterial, which rules that out as the explanation.

**Concrete first target for Phase 3a: the domain-edge bin.** It alone lifts the all-bin rms from 1.64
to 2.72 pt.

---

## What we already know, and what it forces

| finding | source | consequence for this plan |
|---|---|---|
| Attribution is **axis-dependent**: flux -> FLOW (corr +0.93), ~~size -> EMULATOR (3.1x too flat)~~, neighbour flux -> **BOTH, anti-correlated** | 2026-08-01s | there is no single fix; and on the blend axis a partial fix makes things WORSE |
| **The size half of that attribution is WRONG.** On the per-pair ruler the emulator matches truth in every true-size bin (+1.73% ± 3.25 overall); the constgold `r_sim - R_flow` construction had charged the flow's own size error to the blend term | 2026-08-02c | Phase 2 struck; **the size axis belongs to the flow** |
| The emulator's real, ruler-confirmed defect is **separation: −40.9% below 1"** | 2026-08-02c | it is the defect a retune has already failed to move -> Phase 6, not Phase 2b |
| Magnitude-cut bias is the response's **correlation with measured magnitude**, not its mean | 2026-08-01r | a mean-response fix cannot touch it |
| The flow's shear response is added **identically to every draw** | 2026-08-01i/r | no added FEATURE reaches this; the missing dependence is on the measurement REALISATION |
| The RA head was built to fix exactly that and **FAILED** its effect-size floor | 2026-08-01i | a third re-gate is forbidden; both case sets are burned |
| More seeds cannot help (16 checkpoints agree with each other 77x better than with the sim) | 2026-08-01j | no ensemble route |
| Per-bin recalibration **provably cannot** reduce grouping loss, and a non-injective map increases it | Perez-Lebel Lemma C.5 | no post-hoc route |
| Extraction confounder (half-shear forward vs constgold antithetic) is **unremoved** | 2026-08-01s | the attribution above may be wrong; must be closed first |
| Sim additivity `R_total = R_self + R_blend` is **untested**, and required blend goes NEGATIVE at S/N~121 | 2026-08-01s | the whole flow-vs-emulator split rests on it |

---

## Phase 0 — close the two gaps that could invalidate the attribution

**Rationale.** Phases 2-3 spend real effort fixing a named component. If the attribution is
contaminated we fix the wrong one. Both gaps are cheap to close relative to that risk.

**0a. Test sim additivity where `R_blend` is known to be zero.**
For ISOLATED galaxies (`neighbored = False`) `R_blend = 0` by construction, so `R_sim` must equal
`R_flow` exactly, with no emulator involved. Bin that comparison in S/N and true size.
- Clean flow-only measurement on constgold's OWN extraction — this also closes 0b for isolated rows.
- Directly probes the `S/N ~ 121` anomaly (`R_flow = 1.3283` vs `R_sim = 1.3204`, required blend
  `-0.0079`). If the flow over-predicts on isolated high-S/N rows too, that anomaly is a flow error and
  additivity survives. If it does not, additivity is the suspect.
- **Gate:** if isolated-row `|R_flow/R_sim - 1|` is small and flat, additivity is supported and the
  2026-08-01s attribution stands. If it is large and structured, **stop and re-derive the attribution**.

**0b. Measure the extraction difference as a function of true size.**
The half-shear (forward `0 -> +g`) and constgold (antithetic `+-g`) sims are image-identical
(byte-identical `noise_info`), and the extraction gap is already known to be real at the faint end
(`+0.49` vs `+0.60`). What is unknown is whether it is SIZE-dependent — which is exactly what would
contaminate the size-axis attribution.
- **Gate:** flat in true size -> attribution stands. Size-dependent -> the "flow explains 2.07 /
  emulator remainder 2.41" split must be recomputed with the extraction term removed.

**0c. Keep-fraction-matched size comparison** (owed from 2026-08-01m). Compare selection and response
at matched KEEP FRACTION rather than matched threshold — true `Re > 0.31"` against measured `0.60"`,
or a measured cut near `0.92"` against true `Re > 0.6"`. Cheap; removes a caveat currently attached to
every size statement.

**0d. Is the both-detected leg matching injecting bias PER BIN?** (added 2026-08-02 from the
literature review.) Sheldon et al. 2020 §4.3 warn that matching detection lists across sheared images
"would introduce the very shear-dependent object detection biases we wish to calibrate" — and SBSI's
both-detected requirement IS that operation.

This is **not** news to the project: the effect was already measured globally (2026-07 stage-3 work,
`R_full/R_both - 1` = `-0.88%` ALL, `-0.08%` ISO, `-1.11%` BLENDED) and recorded as a correction NOT
present in the certified both-detected `m`. **But it was measured globally, and this entire plan exists
because a global number is not enough.** The blend-dominated split (`-1.11%` vs `-0.08%`) already hints
it is strongly property-dependent.

- Recompute `R_full/R_both - 1` **per bin** in true size, true magnitude and neighbour flux.
- **Why it gates:** if it is large and structured in true size, then part of the size-axis residual
  Phase 2 attributes to the emulator is actually leg-matching bias, and the emulator retune would be
  chasing the wrong target. At `-1.11%` on blended objects against a size-axis rms of `2.82` pt, this
  is not a small enough term to assume away.

**Cost:** four CPU-mostly jobs, hours not days. **Phase 0 gates everything below.**

---

## Phase 1 — make the per-bin number the acceptance metric

**Rationale.** We are currently optimising and reporting a quantity the literature says is
insufficient (MacCrann 2022: global `m` is only the normalisation of `n_gamma(z)`; Kull & Flach: it is
rung 1 of a 3-rung hierarchy). Nothing downstream is safe until the gate changes.

**1a. Promote per-bin residuals to the acceptance metric.** Report, alongside global `m`:
per-bin `m` in true size / true magnitude / neighbour flux, its **rms and max |m|**, and `m` at each
realistic cut. Amend `AGENTS.md` and `CONVENTIONS.md` so the `|m| <= 0.3%` deliverable is stated
per-bin (or at minimum at the realistic cuts), never globally alone.

**1b. State the target honestly.** Current per-bin rms is `2.82` pt on the size axis against a `0.3` pt
budget — roughly an order of magnitude. Setting a per-bin target of 0.3% immediately would be
aspirational; propose an explicit interim (e.g. per-bin rms < 1.0 pt, max |m| < 1.5%) so progress is
measurable and the gate is not permanently red.

**1c. Do NOT invent a grouping-loss number yet.** `LITERATURE_resolution.md` §2.9 gap 1 and 4: grouping
loss is defined only for classification and only on predicted VALUES, not on a response. The per-bin
residual rms we already compute is the honest available statistic. Formalising a grouping loss on a
response is a research contribution, not a step in this plan.

**Cost:** documentation plus a reporting change to existing scripts. No new jobs.

---

## Phase 2 — the emulator — ~~owns the size axis~~ **STRUCK 2026-08-02, premise refuted**

**This phase is retained only as a record of a wrong turn. Do not execute 2b.**

**Why it is struck.** Its target was: the emulator's `R_blend` spans `0.116-0.145` across true-size
bins while the required value spans `0.071-0.160` — "3.1x too flat". That number came from
`required_blend = r_sim - R_flow` on constgold, which (i) assumes additivity, (ii) charges 100% of the
flow's own size error to the blend term, and (iii) uses a target carrying a size-structured
leg-matching bias (0d: 0.93 pt of it). Measured directly on the **per-pair ruler** — the instrument
AGENTS.md requires for emulator claims, with matched extraction, no additivity assumption and no flow
involved — the emulator agrees with truth in **every true-size bin within ~1.9σ** (overall
`+1.73% ± 3.25`; within ~1σ at a 3" aperture), and ruler truth shows no significant size trend either.
The constgold inference would need a variation of ≈77% of the emulator's mean prediction; the ruler
excludes that at many σ.

**What survives, and where it goes.** The ruler confirms a real emulator defect on a DIFFERENT axis:
**−40.9% below 1" separation** (−8.7% at 1-2", −2.1% at 2-3"). That is the known Gold-V3 deficit,
reproduced at the fiducial tuned tag. Gold-V3 already records that it resisted in-domain retraining
(−40.3%) and close-pair loss weighting (−37.3%/−37.1%, saturating), diagnosing "a representational
limit, not an incentive one". **So the surviving defect is exactly the one a retune has already failed
to move, which routes it to Phase 6, not to a Phase 2b retune.**

**The size axis now points at the FLOW**, so it is Phase 3a's problem, not the emulator's.

<details><summary>Original Phase 2 text, kept for audit</summary>

**Target:** the emulator's `R_blend` spans `0.116-0.145` across true-size bins while the required value
spans `0.071-0.160` — **3.1x too flat**, 24% too low at `Re ~ 0.93`, 71% too high at `Re ~ 1.41`.

**2a. Diagnose before retuning: is this a representativeness failure?** Compare the emulator's training
distribution in true size against the population it is applied to. Kannawadi et al. 2019 is the
precedent — a calibration sample whose joint property distribution differs from the target produces
exactly this cancel-globally-fail-per-bin signature, and reweighting does NOT recover it. If the
training set is unrepresentative in size, fix that first; retuning on an unrepresentative set will
just relocate the cancellation point (Byrd & Lipton 2019 on why weighting is not a reliable fix).

**WARNING BEFORE INVESTING IN 2b — the retune precedent is bad.** `Gold-V3.md` records that BlendEMU
under-predicts the per-pair blend response by **-41.5% below 1"** (5.7 sigma, held-out -44.6%), and
that **neither in-domain retraining (-40.3%) nor close-pair loss weighting (-37.3% at 4x, -37.1% at
13x, saturating while degrading 1-2") moved it** — diagnosed there as "a representational limit, not
an incentive one". A size-stratified retune is the same class of intervention on the same model.

The two deficits are on DIFFERENT axes — that one is close-pair separation, ours is true-size
flatness — so the precedent is a warning rather than a verdict. But it means **2b should be time-boxed
and abandoned quickly if it does not move**, with Phase 6 as the real answer.

**2b. Retune with a size-stratified objective — on the RULER, never on constgold.**
`AGENTS.md`: emulator promotion is argued on the per-pair ruler (`scripts/eval_rblend_gap.py`),
**never** on constgold `m`. Constgold is evaluation-only (the R_blend firewall). So the objective gains
a per-size-bin term computed on the ruler; constgold per-bin residuals are read only afterwards, as
evaluation.

**2c. Gate.** Required-vs-predicted `R_blend` per size bin (the `emu/req` column already produced by
`scripts/eval_blend_vs_flow_perbin.py`) plus the existing ruler. **Phase 4 gate applies too.**

**Risk:** the firewall makes this harder than it looks — we cannot optimise directly against the
quantity we care about. If the ruler turns out not to be sensitive to the size-dependence, say so and
stop rather than quietly switching to constgold.

</details>

**Postscript on that last risk line.** It anticipated the right failure mode but guessed the wrong
direction: the worry was that the ruler would be *insensitive* to the size-dependence. It is amply
sensitive — it simply shows the size-dependence is not there. The rule it states ("say so and stop
rather than quietly switching to constgold") is what was followed.

---

## Phase 3 — the flow (owns the flux axis; and the joint defect)

Two genuinely different problems. Do not conflate them.

**3a. Mean-response error vs S/N -> derivative supervision.**
`LITERATURE_resolution.md` §2.7: if the target IS a derivative and the loss trains on values, the loss
is nearly flat along directions that change the derivative while preserving the marginal fit. Sobolev
training (Czarnecki et al. 2017) supervises the derivative directly.

**We already have both halves of this.** The derivative targets are the finite difference between
shear legs. And the machinery has a precedent in this repo: the **theta-coupling pin (`lam_theta=500`)
that fixed the flux/size response** is a derivative-supervision term in all but name. Extending the
same idea to the SHAPE response's dependence on size and S/N is the natural next step, not a new
technique.

- Pilot small (few seeds, short training) before committing to a 16-seed run.
- **Gate:** per-bin residual on the flux axis, AND the flux/size responses must not regress — the pin
  was won once and must not be traded away.
- Honest caveat: no published application of Sobolev training to conditional normalizing flows. This
  is an extension.

**3b. The realisation / joint defect -> do NOT attack it again architecturally.**
The response is added identically to every draw; the RA head was built for this and failed its
effect-size floor; a third re-gate is forbidden. The literature offers no method either
(`LITERATURE_resolution.md` §2.9 gap 3: conditioning collapse is undocumented for flows).

**Recommended response: stop trying to make MEASURED cuts work, and cut on TRUE properties.**
- Our true-cut rows have zero selection bias by construction (`pure_sel` and `keep_offset` exactly
  `0.0000`).
- Euclid independently recommends it: "defining true input bins is also essential to minimise the
  impact of selection bias and not to misinterpret results" (Congedo et al. 2024).
- This converts an unsolved modelling problem into a stated analysis-choice limitation. **It must be
  written down as a limitation, not quietly adopted.**

---

## Phase 4 — the joint gate on the blend axis (non-negotiable)

On the neighbour-flux axis the flow and emulator errors are **anti-correlated** (`-0.436`) and
currently cancel: total residual `1.51` pt against components of `4.15` and `5.08`. Under additivity,
**fixing the flow alone takes that axis from 1.51 to ~5.08; fixing the emulator alone, to ~4.15.**

**Therefore:** no Phase 2 or Phase 3 change may be promoted on its own axis alone. Every candidate is
evaluated on **all three axes**, and a promotion that improves one axis while degrading the blend axis
beyond its current `1.51` pt is rejected or held until its counterpart lands.

This is the single most likely way this plan goes wrong, which is why it is its own phase.

**Reinforced by 0d (2026-08-02).** On that same neighbour-flux axis the both-detected leg-matching
bias runs **−0.15% → −3.12% (span 3.06 pt) — larger than the 1.51 pt total residual it is supposed to
be judged against.** So the blend-axis cancellation is not merely a cancellation between two model
components; it is a cancellation measured against a TARGET that carries its own ~3 pt of structure on
the same axis. Any Phase 4 verdict on this axis that ignores the leg-matching term is judging model
error against a moving ruler. **Report the blend-axis residual alongside the 0d det-bias curve, never
alone.**

---

## Phase 5 — validation that can actually see the failure

**5a. Local, not global, diagnostics.** SBC and expected-coverage average over the prior and will pass
while per-bin structure is wrong (Modrak et al. 2023 documents the blind spot). Use LCT/ALP
(arXiv:2102.10473) or L-C2ST (arXiv:2306.03580), read as functions of true size, magnitude and blend
state. Nearest in-field precedent: Cal-PIT / LADaR (Dey et al. 2025) in photometric redshifts.

**5b. Multiaccuracy audit over the CUT SETS.** A cut on `Z` reports `E[Y - S | Z in A]`, non-zero
exactly when the residual correlates with the indicator — the multiaccuracy condition, much weaker and
cheaper than full multicalibration. Our cuts are known indicator sets, so this is the right and
affordable target. Continuous-case entry point: Globus-Harris et al. 2023, which needs only a
squared-error regression oracle.

---

## Phase 6 — Gold-V3: learn `R_blend` with a flow (owner's proposal, 2026-08-02)

**This already exists as a design.** `Gold-V3.md` ("Folding R_blend into the flow", DESIGN, updated
2026-07-28, nothing built) specifies flow #2 as a firewalled drop-in replacement for BlendEMU:

```
p(measured e | primary true props, ONE true neighbour, separation, flux shells near/mid/far)
```

used only through its neighbours-only derivative (shear the neighbour, hold the primary fixed, read
the mean-head shift), summed over neighbours at evaluation. Its decision was **TWO flows, not one —
for now**, on de-risking grounds: a component swap inside the existing additive frame, so a failure
regresses nothing. Merge deferred to Direction B, where `I = Var(s)` needs a cross term two separate
densities cannot produce.

**Do not re-derive that document. Four findings from 2026-08-01/02 bear on it, and one changes its
central decision.**

**6a. A SECOND failure axis for BlendEMU, strengthening the representational-limit case.** Gold-V3's
motivation is the `-41.5%` close-pair deficit below 1". We have now measured an independent one:
`R_blend` is **3.1x too flat in TRUE SIZE** (spans 0.116-0.145 where the required value spans
0.071-0.160; 24% low at `Re ~ 0.93`, 71% high at `Re ~ 1.41`), while being right to **0.8% globally**.
Two unrelated axes of failure in one model is a stronger argument for replacement than either alone.

**6b. THE TWO-VERSUS-ONE DECISION SHOULD NOW BE MADE BY PHASE 0a, NOT DEFERRED.** Gold-V3 lists
"linearity is BlendEMU's assumption, not a theorem" as a risk. Phase 0a tests exactly that, and the
negative required blend at `S/N ~ 121` is a live hint it may fail. **Two flows still assume
`R_total = R_self + R_blend`** — they only swap what supplies the second term. If additivity fails,
two flows inherit the failure and only an integrated model escapes it. So Phase 0a's outcome should
decide two-vs-one, rather than leaving it to Direction B.

**6c. The blend-axis anti-correlation is an independent argument for integration — with a real cost.**
On the neighbour-flux axis the flow and emulator errors are anti-correlated (`-0.436`) and currently
cancel (total `1.51` pt against components of `4.15` and `5.08`). A single model trained on the TOTAL
response has no decomposition to get wrong and cannot hide error this way. **The cost is that we lose
the diagnostic** — the flow-versus-emulator split is what produced almost every result of
2026-08-01/02, and an integrated model is opaque to it. If we integrate, Phase 1's per-bin metric and
Phase 5's local diagnostics become the ONLY instruments left, which raises their priority rather than
lowering it.

**6d. Flow #2 would INHERIT the realisation defect if built like flow #1.** Gold-V3 predates
2026-08-01r. The response living entirely in the mean head `_mu(c)`, added identically to every draw,
is an architecture property — a second flow of the same construction reproduces it, and would then
carry it into the blend term as well. **This must be designed against explicitly, not discovered
afterwards.** Note Gold-V3's own spec is "used only through its mean-head shift", which is precisely
the construction that has no realisation structure.

**6e. Phase 0d and Gold-V3's outstanding test 3 are THE SAME INVESTIGATION.** Gold-V3's leading
surviving explanation for the `-41%` is close-pair **detection selection**: "our truth sample at
sub-arcsecond separation is conditioned on both objects being detected... the population we average
over is therefore not the one the emulator was trained on, which *does* bias a conditional mean."
That is the same mechanism as Sheldon et al. 2020's leg-matching warning behind Phase 0d. **Run it
once, binned by separation AND by true size**, and it serves both.

**Sequencing.** Phase 6 is the structural answer and Phase 2b the cheap one; given the retune
precedent, expect to reach 6. Phase 0a gates the two-vs-one design choice. Gold-V3's own outstanding
tests 4-6 (primary-shape/pair-axis angle, does one neighbour carry it, multiplicity-binned held-out
NLL) stand unchanged.

---

## Explicitly NOT doing, and why

| ruled out | reason |
|---|---|
| per-bin recalibration of `m` | Perez-Lebel Lemma C.5 — provably cannot reduce grouping loss; non-injective maps increase it. Also collides with AGENTS.md "no silent empirical corrections" |
| more seeds | 2026-08-01j — all 16 checkpoints agree with each other 77x better than any agrees with the sim. Structural, not initialisation noise |
| a third RA re-gate | 2026-08-01i — both case sets burned; explicitly forbidden |
| adding conditioning features to fix the measured-cut bias | the missing dependence is on the measurement REALISATION, not on any feature we could add |
| tuning the emulator against constgold `m` | R_blend firewall; constgold is evaluation-only |
| importance-weighting the calibration set to the target population | does not change the pointwise error at all — it relocates the cancellation point (and Byrd & Lipton 2019: the effect can vanish entirely in overparameterised nets) |

---

## Sequencing

```
Phase 0  RUN 2026-08-02:  0a BLOCKED | 0b non-detection | 0d real but ~9% of the size span
   |                      0c still owed
   |     and, outside the four gates, the ruler test that STRUCK Phase 2
   |
Phase 1  (per-bin acceptance metric)                   <- GATES 3, 6.  Unaffected, unblocked
   |
   +---- Phase 2  ** STRUCK ** premise refuted on the ruler; the surviving defect is
   |              separation, which Gold-V3 already showed resists retuning -> Phase 6
   |                                                                             |
   +---- Phase 3a (flow, derivative supervision)  <- NOW OWNS BOTH flux AND size -+--> Phase 4
   |                                                                             |    joint
   +---- Phase 6 (Gold-V3: learn R_blend with a flow) ---------------------------+    blend-axis
   |                                                                                  gate, now
   +---- Phase 3b (true-property cuts; a documented limitation, not a fix)            read WITH
                                                                                      the 0d curve
                                                                                        |
                                                                                        v
                                                                                     Phase 5
```

Phases 3a and 6 are independent and can run in parallel, but **neither may be promoted without the
Phase 4 joint gate** — which must now be read alongside the 0d leg-matching curve, since on the blend
axis that curve (3.06 pt) is larger than the residual being judged (1.51 pt).

**Phase 3a is now the main line.** With Phase 2 struck, the size axis joins the flux axis as the
flow's problem, so derivative supervision has two axes to answer for rather than one — which raises
both its value and its risk. Its gate is unchanged: the `lam_theta` flux/size pin must not regress.

**If Phase 6 lands, Phase 4's gate does not disappear — it changes shape.** An integrated model has no
flow-versus-emulator split to check, so the blend-axis guard becomes a per-bin residual on the
neighbour-flux axis alone, and Phases 1 and 5 become the only instruments. Their priority goes up,
not down.

## What would make me abandon this plan — **scored 2026-08-02**

- ~~**Phase 0a shows additivity fails.**~~ **Not answered — the test is BLOCKED**, no isolated
  subpopulation exists (0.03% of rows, ±11.1%). Additivity remains untested, so every statement below
  that rests on `R_total = R_self + R_blend` is still resting on it. Note MacCrann et al. 2022 concede
  the field cannot cleanly decouple detection from neighbour contamination either.
- ~~**Phase 0b shows the extraction difference is size-dependent.**~~ **Not detected** (χ²/dof 2.2/5),
  but with a weak bound. The size-axis attribution is not invalidated by extraction.
- ~~**Phase 2a shows the emulator's training set is representative.**~~ **Moot — Phase 2 is struck.**
  The premise it was to diagnose (true-size flatness) is not visible on the ruler at all.
- ~~**Phase 0d shows the leg-matching bias is large and size-structured.**~~ **Partly.** It is real and
  significant but nearly a constant offset in size (span 0.93 pt, ~9% of the residual span), so it
  does not overturn the size-axis attribution on its own. **It IS large on the blend axis (3.06 pt,
  above the 1.51 pt residual)** — that moves to Phase 4, not to abandonment.

**Net: the plan is not abandoned, but its centre of gravity moved from the emulator to the flow.**
The one thing that did fall was Phase 2, and it fell to a direct measurement rather than to any of the
four criteria written above — which is worth noting, because none of the pre-registered abandonment
criteria caught it. The criterion that would have is "test the premise on the firewall-clean
instrument before building a phase on an inferred quantity", and it is now the first thing Phase 0
should do for any future phase.

---

## Provenance of each phase

Recorded so the plan's own reasoning can be audited later. See `LITERATURE_resolution.md`.

| item | origin |
|---|---|
| Phase 0a, 0b, 0c | **ours** — own measurements and caveats; no literature analogue |
| Phase 0d | **literature** (Sheldon et al. 2020 §4.3) applied to a bias this project had already measured globally |
| Phase 1 | **ours**, independently confirmed by MacCrann 2022 and Kull & Flach 2015 |
| Phase 2a ordering (diagnose representativeness before retuning) | **literature** (Kannawadi et al. 2019) |
| Phase 2b, 2c | **ours** (the firewall is an AGENTS.md rule) |
| Phase 3a derivative supervision | **literature** (Czarnecki et al. 2017) naming and generalising a technique already used here as the `lam_theta` pin |
| Phase 3b true-property cuts | **ours**, independently confirmed by Euclid/Congedo et al. 2024 |
| Phase 4 blend-axis joint gate | **ours** — the anti-correlation is our own measurement; the literature has nothing on this |
| Phase 5a local diagnostics | **literature** (Zhao et al. 2021, Linhart et al. 2023, Modrak et al. 2023) |
| Phase 5b multiaccuracy over cut sets | **literature** (Hebert-Johnson 2018, Kim 2019, Globus-Harris 2023) |
| Phase 6 | **owner's proposal, and already designed** in `Gold-V3.md` (2026-07-28). 6a-6e are new evidence bearing on it from 2026-08-01/02, of which **6b changes its central two-flows-vs-one decision** |
| NOT-DOING: per-bin recalibration | **literature** (Perez-Lebel Lemma C.5) — a proof, replacing a hunch |
| NOT-DOING: importance weighting | **literature** (Byrd & Lipton 2019) |
| NOT-DOING: more seeds, third RA re-gate, more features, constgold tuning | **ours** |

**The load-bearing gates (0 and 4) are ours. The literature supplied techniques and prohibitions
inside a structure it did not provide.**

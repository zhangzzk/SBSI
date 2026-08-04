# SBSI Conventions — definitions, setups, and which one to use when

Purpose: make any two SBSI tests comparable by fixing the vocabulary and the setup. If a number in a
figure, table or WORKLOG entry does not say which of these it used, it is under-specified.

**Scope.** This file defines *how quantities are built*. It does not restate the fiducial MODEL, the
seed convention, or the numerical-integrity rules — those live in `AGENTS.md` and are referenced from
here. Where the two disagree, `AGENTS.md` wins and this file is the bug.

**Status of each claim below.** Everything is stated from the code as of 2026-07-31 and the file/line
is given where it is short enough to check. Section 7 (detection) is the one area where this file
deliberately points elsewhere rather than restating an estimator — see the note there.

---

## 1. The model

Fiducial = **V2 dom6x6 flow + tuned in-domain blend emulator**, and the prediction is always

```
R_model = R_flow + R_blend
```

- `R_flow` — the flow's own response. **SELF-response only.** It is never the whole model, and a
  table whose model column is `R_flow` alone is missing the neighbour term (this produced a constant
  −15.32% offset that was misread as model error).
- `R_blend` — per-object, from the emulator lookup, **averaged over the same object set as the
  shapes**. Under a cut it must be re-averaged over the objects that pass, not carried over from the
  no-cut population.

Checkpoints, emulator tag, lookup path and the two silent traps: `AGENTS.md` → "Fiducial Model".

---

## 2. Catalogues

Two families. **They are not interchangeable and must not be described with the same words.**

### 2a. constgold — the CONSTANT-shear, antithetic catalogue

```
/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/
    constant_response_catalogue_train.feather        --min-case 40
```

- **Antithetic**: each galaxy is rendered at `+g` and `−g` with the **same noise**, opposite shear.
- One row per galaxy **detected in BOTH signs** (`blendemu/response.py`, `detec_p`/`detec_m` from the
  CrossMatch). Both-detected is baked into the catalogue — you do not opt into it.
- Per-leg columns are `*_plus` / `*_minus`: `measured_e1/e2`, `S/N`, `measured_mag_auto`,
  `measured_flux_radius`.
- ngmix failures are **dropped**, not kept as NaN (fixed 2026-07-31; the older builder kept ~5,300
  such rows and the certified file did not).
- Cases 40–139 are the evaluation split. The `c40-*` split files are STALE old-centroid — see
  `reference_constgold_catalogue` memory.

**constgold is EVALUATION ONLY.** Nothing may be trained on it and no model or hyperparameter may be
selected using its `m`. This is the R_blend firewall.

### 2b. half-shear legs — the FORWARD catalogue

```
det_meas_ngmix_g0.0_train.feather     (leg 0: g = 0)
det_meas_ngmix_g0.05_val.feather      (leg g: |g| = 0.05)          cases <= 39
```

- **Forward**, not antithetic: leg 0 is `g=0` and leg g is `+g`. Legs are separate files.
- Constant shear MAGNITUDE, **random direction per galaxy** (`shear_angle`). This is what lets the
  self-response be isolated — see §6b.
- Do **not** call these "constant-shear"; that name belongs to constgold.

---

## 3. Building the population — order matters

The standard order, as in `scripts/eval_selection_constgold_neardomain.py`:

1. **Case cut** — `case >= 40` (constgold) or `case <= 39` (half-shear).
2. **Quality / selection cut** — `source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)`.
   All on **TRUE** properties (`sbs_shear/preprocessing.py:108-118`):
   - `18 < r_input_p < 28`
   - `0.1 < Re_input_p < 1.5`
   - `(0 < distance < 5)` **OR** `not neighbored`
3. **Domain cut** — `r_input_p < 26` **and** `Re_input_p > 0.3`, on TRUE properties. **Mandatory**
   whenever a dom6x6 flow is scored: the flow was trained in this box, and scoring outside it put
   `R_model(no cut)` at +0.173 against ~0.29 and inflated every model entry ~4x (run 15365425).
4. **Emulator-lookup join** — inner join on `(case, input_index)`. Rows with no `R_blend` are
   **DROPPED, never zero-filled**, and the match fraction is asserted. Zero-filling collapsed
   `<R_blend>` 0.159 → 0.059 and produced a spurious +28.9% m (job 15366950).
5. **Finiteness mask** — every per-leg quantity used must be finite in BOTH legs.

Steps 2–4 overlap on purpose: after step 4 the emulator's own box (mag 18–26, Re 0.3–1.5) already
subsumes step 2's ranges, so step 2 removes nothing in-domain (measured: 0.00%). That is a
consistency check, not redundancy to delete.

The resulting standard in-domain population is **N = 11,674,408**.

**Steps 1–4 are TRUE-property cuts. They define the population.** They are NOT the selection effect
being measured — see §5.

---

## 4. Which shape to use

Three "kinds" are carried side by side. Each answers a different question; mixing them silently is
the most common way to get a wrong number.

| kind | what it is | use it for |
|---|---|---|
| `unsheared` | the raw intrinsic shape, **projected identically in both legs** | isolating the moving-boundary term (§5) |
| `sheared` | the intrinsic shape with the shear applied analytically | the ideal/noiseless reference; its response is ~1.000 by construction |
| `measured` | the ngmix shape from the image | **the sim truth for any `m`** |

`unsheared` has response identically **0** with no cut, because the same array is fed to both legs.
Anything nonzero under a cut is therefore purely the boundary moving — which is exactly what makes it
the pure-selection probe.

Shapes are always in the SBS canonical spin-2 basis `(q1,q2) = q(cos 2θ, sin 2θ)`.

---

## 5. Cuts: true vs measured — the distinction that decides whether there is any selection effect

- **TRUE-property cuts do not move with shear.** The same galaxies pass in both legs, so the
  selection term is **zero by construction**. Use them to define a population (§3), never to measure
  a selection effect.
- **MEASURED cuts move with shear.** Measured mag/size/S-N depend on the applied shear, so the
  boundary shifts between legs and galaxies scatter across it in a shear-correlated way. **All
  selection results must use measured cuts.**

Apply the measured cut **per leg**: build `pass_plus` from the `+` leg's measured values and
`pass_minus` from the `−` leg's. Applying one leg's mask to both destroys the effect being measured.

**"No cut" means no further MEASURED cut** — the population is still TRUE-cut per §3. It is not an
unselected catalogue.

Two traps this creates, both real:

- **Measured `mag<26` is NOT a no-op** even though the population is true-mag<26: measurement scatter
  pushes ~2.5% past the boundary, and it carries a genuine selection excess.
- **Measured size cuts below ~0.5″ ARE no-ops**, for an unrelated reason: the PSF (Moffat FWHM 0.73″,
  β=2.224 → R₅₀ = 0.5268″) floors measured `flux_radius`, so nothing lands below 0.30″ whatever its
  true size. `R>0.30"` and `R>0.40"` keep ~100% and are structurally empty. Always report the keep
  fraction so an empty cut cannot be mistaken for a passing one.

---

## 6. Response estimators

### 6a. Two-means leg average (both catalogues)

```
R = ( mean(x[pass_plus]) − mean(x[pass_minus]) ) / (2g)
```

(`leg_avg`, `scripts/eval_selection_constgold.py:64`.) For forward legs the denominator is `g`, not
`2g` — see 6b.

### 6b. Isolating the SELF response (half-shear only)

constgold cannot do this: every object's response there contains its neighbours' contribution.
Half-shear can, because each galaxy carries its own random shear direction `ĝ_p`:

```
R_self = < (e_g − e_0) · ĝ_p > / g
```

Neighbours' `ĝ_s` is uncorrelated with `ĝ_p` and averages away.

### 6c. Extraction convention must MATCH on both sides

The half-shear sim is **forward** (`0 → +g`), so the flow must be scored **forward** too
(`s=0 → s=+g`), never antithetic `±g`. constgold is antithetic, so its model side is antithetic.
Mixing them reintroduces a recorded 0.49-vs-0.60 self-response gap that is pure extraction
convention on image-identical sims — not physics.

### 6d. Common Random Numbers

Reseed the flow identically in both legs (`torch.manual_seed(seed)` before each leg). The difference
is `O(g)` and the raw draws are `O(1)`, so without CRN the sampling noise is amplified ~1/g ≈ 20x
instead of cancelling.

### 6e. The model must re-select, not inherit

When a cut is applied, the model side samples its **own** measured mag/size from the flow, applies
the cut to those, and averages both `R_flow` and `R_blend` over whatever passes. It must **not**
inherit the sim's pass mask — predicting which objects survive is the thing being tested.

---

## 7. Leg matching: both-detected, and when not to

- **Default = matched, both-detected.** constgold is built that way; the half-shear base is an inner
  merge of the two legs on `(case, input_index)` (`eval_selection_response.py:87`).
- **This removes the DETECTION selection effect by construction.** A galaxy detected in one leg and
  not the other is discarded, which is precisely the population whose loss carries detection bias.
  So a both-detected `m` is a shape(+blend) number and is silently missing `R_detect`.
- **Detection-bias work must therefore NOT restrict to both-detected.** The measured size of what
  this omits is ~−0.9% globally and ~−1.1% on blended objects.

**Deliberately not restated here:** the detection-response estimator itself. It is a separate
construction (response-regularised `P(detect | true + blend)`) and lives in the stage-3 work; see the
`project_stage3_detection` memory and the WORKLOG entries it points to. Copying a summary of it into
this file would create a second, drifting definition — which is the failure mode this file exists to
prevent.

---

## 8. `m` and its relatives — one sign convention

**`m = R_sim / R_model − 1`, everywhere.** Same as `AGENTS.md`, same as fig3. Never the inverse.

An earlier near-domain table inverted it for the cut column only while the no-cut reference stayed
`sim/model − 1`. That mirrored the two about zero: a row keeping 99.99975% of the population — i.e.
the no-cut case — printed +0.247% against a no-cut line at −0.245%, turning an exact null into an
apparent 0.49-pt disagreement.

With `R0` = no-cut and `R(cut)` = under the cut:

| name | definition | what it answers |
|---|---|---|
| `m` | `R_sim(cut) / R_model(cut) − 1` | how wrong is the model on this sample |
| **(1) pure selection** | `R_unsheared(cut) / R_sheared(R0)` | the moving-boundary term alone |
| **(3) measured shift** | `R_sim,meas(cut) / R_sim,meas(R0) − 1` | how much the SIM's response moves under the cut |
| **(4) model shift** | `R_model(cut) / R_model(R0) − 1` | the model's counterpart of (3) |
| `dm` | `m(cut) − m(no cut)` | the selection-INDUCED excess |

### The shift (3) contains TWO things

1. **Population re-weighting** — even a shear-independent cut changes which galaxies are averaged, and
   `R` varies strongly with brightness and size. **Not a bias**, but the model must reproduce it.
2. **Moving-boundary selection** — the shear-correlated boundary crossing. This is selection bias.

Column (1) isolates (2), so **(3) − (1) ≈ the re-weighting part**. Worked example: `mag<25` shifts
+16.75% of which only −0.08% is boundary — nearly all re-weighting, because you kept bright
high-response galaxies. `R>0.70"` shifts +11.93% with a +5.44% boundary term.

**Caveat on comparing (1) with (3) quantitatively.** (1) averages INTRINSIC shapes and (3) averages
MEASURED ones, and measured shapes are diluted relative to intrinsic. Renormalising (1) by
`R_meas(R0)` to share (3)'s denominator is only half the conversion — the missing dilution factor
works the other way and the two largely cancel. That factor is unmeasured, so **neither
normalisation is asserted correct**; read (1) against (3) qualitatively. `pure_sel_meas` is stored as
a diagnostic only.

### Errors: form ratios PER SEED

Build each ratio inside each seed, then take the spread across seeds. Forming it from ensemble means
and propagating the numerator's scatter alone discards the seed cancellation and returns the same
`±0.43%` on every row regardless of cut severity — a reliable tell that this bug is present.

Where the seed offset cancels:

- **Cancels** — (4), and `dm`. Both terms carry each seed's offset. Errors come out ~20x tighter.
- **Does NOT cancel** — absolute `m`. It is model-vs-SIM and the sim side has no seed dependence.

Note `mean-of-per-seed-ratios ≠ ratio-of-ensemble-means`. Use mean-of-per-seed, to match how the
per-object dumps build the ensemble `m`. For the no-cut row the two read −0.239% and −0.245%; a −0.245
in an older log is the ratio-of-means version, not a different population.

The seed spread is **not the whole error on `dm`.** The sim side has no seed dependence, so its own
shot noise never enters it, and it does not cancel in `dm` either — a cut and its complement are
disjoint draws. Add `(σ/R)·√((1−f)/(N·f))` with the per-object response scatter σ (5.06 against a
mean 0.86). It is the same order as the seed error and roughly doubles it on aggressive cuts.

### Reading `dm`: two traps that make a real number mean the wrong thing

**Trap 1 — the sign of a same-side cut is FORCED.** `R(no cut)` is the keep-weighted blend of any cut
and its complement, on the sim and model side alike, so `1+m(no cut)` is a weighted mediant of
`1+m(cut)` and `1+m(complement)` and lies strictly between them. Therefore
`sign(dm(cut)) = −sign(dm(complement))`, always. A table whose cuts all keep the bright/large corner
**cannot** produce disagreeing signs, so "positive at every cut" is an algebraic identity, not
evidence. Confirmed by measurement (2026-07-31h): all 7 complements flip
(`mag<26 +0.325 → mag>26 −23.93`, `R>0.70" +4.589 → R<0.70" −27.13`). Run
`--complements` before reading anything into a common sign, and count independent cut DIRECTIONS,
not rows — nested thresholds, no-op rows, and duplicate rows are not independent evidence.

**Trap 2 — `dm` is only a response error if both sides cut the SAME population.** The model applies
the cut to flow-SAMPLED measured quantities; the sim applies it to real measurements. Compare the
`mkeep` / `d%` columns before interpreting:

- `d% ≈ 0` (magnitude rows, −0.03% to −0.13%) → same population; `dm` is a genuine response error.
- `d%` of order a percent (size rows, −0.90% at `R>0.60"`, −1.32% at `R>0.70"`) → **different**
  populations; `dm` mixes a response error with an error in the flow's predicted measured-quantity
  distribution and must not be quoted as a shape bias.
- Keep-fraction-MATCHED rows (the `S/N` proxy rows) have `d% ≈ 0` **by construction**, so their
  column is uninformative — not a passed check.

A fuzzy model-side boundary under-delivers the response shift even when it keeps fewer objects
(at `R>0.70"` the model keeps 0.7817 vs the sim's 0.7921 yet shifts +7.02% against +11.93%), because
noisy selection picks a less extreme subpopulation than a sharp one. Known instance: the flow's
measured `flux_radius` ignores the PSF floor (`R50 = 0.5268"`), putting 19.7× more draws below 0.30"
than the sim has.

---

## 9. Seeds

Split by **which flow OUTPUT drives the reported number**: e (`g1/g2`) response → **16 seeds**;
flux/size → **4**. Ask what the number IS, not what the cut is on — a selection table cuts on
flux/size but reports `m`, a shape-response bias, so it takes 16.

Full statement, rationale and the worked example: `AGENTS.md` → "Ensemble Seed Convention".

---

## 10. Checklist for a new test

1. Which catalogue — constgold (antithetic) or half-shear (forward)? Extraction must match it (§6c).
2. Population built in the §3 order, with the domain cut present if a dom6x6 flow is scored?
3. Emulator match fraction asserted, unmatched rows dropped not zero-filled?
4. Cuts on MEASURED quantities, applied per leg (§5)?
5. Keep fraction reported, so an empty cut is visible?
6. Model side re-selecting on its own sampled measurements (§6e)?
7. `R_blend` re-averaged over the passing set, not carried over?
8. `m = R_sim/R_model − 1`, and the same convention for every column?
9. Errors from per-seed ratios; seed count set by §9?
10. CRN on if legs are differenced (§6d)?
11. Is "both-detected" doing something you did not intend (§7)?
12. Nothing trained or selected on constgold (§2a firewall)?

---

## 11. Provenance

Written 2026-07-31 after a fig1–5 review turned up two convention bugs (mirrored sign, ensemble-mean
errors) and two clarifications (what "no cut" means; the seed split is by flow output). See
`WORKLOG.md` 2026-07-31c and 2026-07-31d.

When a convention changes, edit this file **and** say so in `WORKLOG.md`. A convention that lives only
in a script docstring will drift.

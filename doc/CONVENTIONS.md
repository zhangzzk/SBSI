# SBSI Conventions — definitions, setups, and which one to use when

Purpose: make any two SBSI tests comparable by fixing the vocabulary and the setup. If a number in a
figure, table or WORKLOG entry does not say which of these it used, it is under-specified.

**Scope.** This file defines *how quantities are built*. It does not restate the milestone MODEL
(that is `MILESTONE.md`) or the numerical-integrity rules (those live in `AGENTS.md`) — both are
referenced from here. The seed convention lives in §9 below. Where this file and `AGENTS.md`
disagree, `AGENTS.md` wins and this file is the bug.

**Status of each claim below.** The model and population rules were updated for
the V3 milestone on 2026-08-17. Historical implementation paths mentioned in
this document now live under `archive/pre-v3/`; they are provenance, not API.

---

## 1. The model

For the frozen scientific comparisons, V3 is V2.2 flow plus the latest
narrow-domain emulator; V3.1 is the four-seed original-E mixed-shear flow plus
that exact same emulator; V3.2 keeps both V3.1 components and replaces the
catalogue-inference detector with the transition-aware SBSI classifier; and
V3b is the named broad-domain comparison. These
are model-path presets, not separate software pipelines. For every model
choice, the prediction is always

```
R_model = R_flow + R_blend
```

- `R_flow` — the flow's own response. **SELF-response only.** It is never the whole model, and a
  table whose model column is `R_flow` alone is missing the neighbour term (this produced a constant
  −15.32% offset that was misread as model error).
- `R_blend` — per-object, from the emulator lookup, **averaged over the same object set as the
  shapes**. Under a cut it must be re-averaged over the objects that pass, not carried over from the
  no-cut population.

Exact checkpoint and emulator paths are exposed by the optional
`sbsi.models` presets; see `MILESTONE.md` for the frozen result record.
User-selected catalogues and emulator-response products remain external API
inputs.

V3.2 detection views use every candidate inside 3 arcsec and retain the one
maximizing `flux_secondary * (Re_secondary / distance)**1`.  The classifier is
then evaluated on the sheared primary and retained-neighbour ellipticities.
This neighbour rule is explicit inference configuration: it must not fall back
to the older nearest-neighbour detector convention.

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
- Cases 40–139 are the evaluation split. The `c40-*` split files are STALE old-centroid (quarantine
  record: `Gold-V1.md` §3).

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

### 2c. inference prior — the fresh FS2 catalogue

The default catalogue-prior identity is registered in
`configs/default_catalogue_prior.json`.  Its immutable external manifest is:

```
/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/
    fs2_prior_cases20000_20199_v1/default_prior_manifest.json
```

- FS2-25876 cases 20000--20199, generator seeds 20123--20322, are disjoint
  from all simulation/model-development cases 0--899.
- The prior leg is exactly unsheared.  All 139,936,000 rows remain available
  as neighbour context; the 12,760,990 positive-mass primary atoms satisfy
  the V3.1 truth domain `18<r<25.8, 0.5<Re<1.5`.
- The store is partitioned into twenty ten-case scene shards.  A consumer must
  use every shard with the manifest's `global_shard_mass`; substituting one
  shard is a smaller diagnostic prior, not the default 200-case prior.
- GalSBI scene catalogues are historical sampler/transfer stress tests only.
  They are not the default inference prior and must not be mixed into an FS2
  model result.

---

## 3. Building the population — order matters

The standard order implemented by `ResponsePredictor` is:

1. **Case cut** — `case >= 40` (constgold) or `case <= 39` (half-shear).
2. **Quality / selection cut** — `source_select_selection(df, cuts=DEFAULT_SELECTION_CUTS)`.
   All on **TRUE** properties (`sbsi/preprocessing.py:108-118`):
   - `18 < r_input_p < 28`
   - `0.1 < Re_input_p < 1.5`
   - `(0 < distance < 5)` **OR** `not neighbored`
3. **Registered domain cut** — on TRUE properties and mandatory. V3 uses
   `18 < r_input_p < 25.8` and `0.5 < Re_input_p < 1.5`; V3.1 uses the same
   domain; V3.2 uses the same V3.1 domain; while V3b uses
   `18 < r_input_p < 26` and `0.3 < Re_input_p < 1.5`. Never infer these cuts
   from a filename or substitute one model's domain for another.
4. **Emulator-lookup join** — inner join on `(case, input_index)`. Rows with no `R_blend` are
   **DROPPED, never zero-filled**, and the match fraction is asserted. Zero-filling collapsed
   `<R_blend>` 0.159 → 0.059 and produced a spurious +28.9% m (job 15366950).
5. **Finiteness mask** — every per-leg quantity used must be finite in BOTH legs.

Steps 2–4 overlap on purpose. The registered flow and emulator must be scored
on their common named domain; overlap is a consistency check, not redundancy
to remove.

The frozen populations are **N = 5,642,349 for V3, V3.1, and V3.2** and
**N = 11,674,408 for V3b**.

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

The archived pre-V3 implementation called this helper `leg_avg`. For forward legs the denominator is `g`, not
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
construction (response-regularised `P(detect | true + blend)`) and lives in the stage-3 WORKLOG
entries. Copying a summary of it into
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

Full statement, rationale and the worked example below, restored 2026-08-18 from the
"Ensemble Seed Convention" section that lived in `AGENTS.md` until the 3193cc5 restructure
(and survived only in git history after that):

**The split is by WHICH FLOW OUTPUT drives the reported number, not by shape-vs-selection**
(clarified by the owner 2026-07-31; the earlier "shape 16 / selection 4" wording was too loose and
led to a 4-seed fig4 that should have been 16):

- **e (`measured_ngmix_g1/g2`) response → 16 seeds.** Any multiplicative bias `m`, the certified
  pipeline number, any absolute shear response.
- **flux / size outputs → 4 seeds.** Flux- and size-response validation
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

---

## 10. Checklist for a new test

1. Which catalogue — constgold (antithetic) or half-shear (forward)? Extraction must match it (§6c).
2. Population built in the §3 order with the selected model's checkpoint domain?
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

Written 2026-07-31 after a fig1–5 review and updated 2026-08-17 for V3/V3b.
See `WORKLOG.md` for the corresponding provenance.

When a convention changes, edit this file **and** say so in `WORKLOG.md`. A convention that lives only
in a script docstring will drift.

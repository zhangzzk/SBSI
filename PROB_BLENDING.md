# Probabilistic Blending — Forward-Modelling the Neighbour Distribution

**Status:** design + prototype (started 2026-07-09, autonomous session)
**Goal:** remove the "true neighbours provided" idealisation from `R_blend`. In real
data we only have the *detected* catalogue with *measured* properties; the neighbour
population that drives the coherent blend response must be **forward-modelled**, and its
undetected part is set by the **classifier** (selection informed by blending).

---

## 1. Where the idealisation lives today

The SBSI coherent shear response of the selected sample decomposes as

```
R_total = R_flow + R_blend (+ R_sel)
```

- `R_flow`  — self-response: the object's own shape reacting to its own shear (measurement flow).
- `R_blend` — coherent blend response: the object's measured tangential ellipticity reacting
  to the shear of its **neighbours** (flux contamination, centroid/deblend coupling).
- `R_sel`   — selection response: objects entering/leaving the sample with shear (classifier).

`R_blend` is produced by the XGBoost pair emulator `f_reg` fed the **true input field** as both
primary and secondary (`build_blend_lookup.py`, `predict_response(field, field)`):

```
R_blend(i) = Σ_{j : true nbr of i, θ_ij < R_max}  f_reg(x_i, x_j, θ_ij)
```

with `x = (Re, r, sersic_n)` the **true** galaxy properties and `θ_ij` the separation. Two
idealisations are baked in: (a) neighbours include **undetected** galaxies we could never see,
and (b) their properties are **true**, not measured. In practice we have only the detected
catalogue `O = {ô_i}` with measured `x̂_i`.

## 2. The probabilistic replacement

Marginalise the blend response over the *posterior* neighbour configuration `N` of each
observed object:

```
R_blend(ô_i) = E_{ N ~ p(N | ô_i, O) } [ Σ_{j∈N} f_reg(x_i, x_j, θ_ij) ]
```

Because `f_reg` is **additive over neighbours**, the expectation of the sum is the sum of the
expectation — we never need to draw full scenes, only the neighbour *intensity*.

## 3. Detected vs undetected split

`N = N_det ∪ N_undet`:

- **`N_det`** — neighbours that were themselves detected ⇒ already in `O` with measured
  properties. Known positions, noisy properties. To leading order directly computable:
  ```
  R_blend^det(ô_i) = Σ_{j∈O, θ_ij<R_max} f_reg(x̂_i, x̂_j, θ_ij)
  ```
  (residual = measurement scatter `x_j ← x̂_j`, already the flow's domain.)

- **`N_undet`** — neighbours below detection ⇒ NOT in `O`. Must be forward-modelled from the
  population. **This is the genuinely probabilistic piece.**

## 4. The undetected-neighbour model = population × (1 − classifier)

Undetected galaxies are a **thinned point process**. Let `Φ(m,s,n)` be the parent property
function (surface density per arcsec² per `dm ds dn`) — the deep luminosity/size/index function,
taken from deep truth or an external deep field. A galaxy of properties `(m,s,n)` at separation
`θ` from object `i` is *undetected* with probability `1 − p_det`, and **`p_det` is exactly the
trained classifier `bst_cla`** — this is the "selection informed by blending" link. The
undetected-neighbour intensity around object `i` is

```
λ_undet(θ,m,s,n | i) = Φ(m,s,n) · [1 − p_det(m,s,n,θ ; i)] · ξ(θ,m | i)
```

(`ξ` = clustering enhancement over Poisson; `ξ≡1` for the first pass). By Campbell's theorem
the expectation over the point process is a deterministic **intensity integral** — no sampling:

```
R_blend^undet(ô_i) = ∫ 2πθ dθ ∫ dm ds dn  λ_undet(θ,m,s,n|i) · f_reg(x̂_i ; m,s,n,θ)
```

**The only new ingredients** beyond the existing framework are `Φ(m,s,n)` (a population prior)
and reuse of the **already-trained** classifier as the thinning kernel. `f_reg` is unchanged;
no emulator is retrained.

## 5. Second-order: conditioning (the Bayesian tightening)

The intensity above is the *field-average* undetected population. Strictly, conditioning on
"object `i` was **detected** and measured as `x̂_i`" reweights the neighbour posterior:

```
p(N_undet | ô_i) ∝ p(x̂_i, detected | x_i, N_undet) · p(N_undet) · p(x_i)
```

The likelihood `p(detected|·)` is again the classifier; `p(x̂_i|x_i,N)` is the measurement
model. For the **mean** response the field-average intensity is the zeroth-order term and the
conditioning correction is second order (a neighbour both shifts detection *and* adds response).
Build the zeroth-order integral first; add conditioning only if 0.3% demands it.

## 6. Validation ladder

1. **Characterise** (`probblend_characterize.py`): split truth-fed `R_blend` over detected
   primaries into detected-neighbour vs undetected-neighbour parts. → how big is the piece the
   forward model must supply, and the bias from dropping it (`Δm ~ <R_blend^undet>/R_total`).
2. **Reconstruct**: `R_blend^det + R_blend^undet(field-avg)` vs truth-fed `R_blend`, per object
   and in the mean.
3. **Propagate**: feed the forward-modelled `R_blend` through the constant-shear coherent
   validation — does `|m| < 0.3%` survive without the true neighbour catalogue?

## 7. Open choices / risks

- `Φ(m,s,n)`: from the sim's own deep input LF (available truth) for the controlled test;
  in production from an external deep field (COSMOS-like). Depth mismatch → residual.
- Clustering `ξ`: neighbours are *more* correlated than Poisson; ignoring `ξ` under-counts close
  undetected pairs, exactly where `f_reg` is largest. Measure the pair correlation from truth.
- Classifier domain: `p_det` must be reliable near the detection limit and at small `θ`, where
  it decides the undetected census. Check its calibration there.

---

## 8. Results (2026-07-10, main set fs2_25876, tag `lsst_r_extnbr_ho`, 4 cases)

**The stakes.** Undetected neighbours carry 7.8% of `R_blend`; dropping them ⇒ **Δm = +3.29%** —
the dominant blending systematic.

**The classifier is a faithful thinning kernel.** Soft `Σ(1−p_det)f_reg` reproduces the hard
detection-truth census to **8%** (`undet_soft/undet_hard = 1.084`).

**The field is Poisson.** Measured ξ(θ) ≈ 1 at all separations (the sim places galaxies at random),
so the population intensity is pure Poisson here. (Real data has clustering → fold in the measured
`w(θ)`; the code has the ξ(θ) hook.)

**The population forward model works.** With NO true neighbour positions — field LF `Φ` +
classifier `p_det` + emulator `f_reg` — the undetected term is reconstructed via:
1. neighbour props ~ `Φ`, separations area-uniform, expected count `ρπ(R_max²−θ_min²)`;
2. neighbour detection thinned by `(1−p_det)` with the **icat2cla-faithful context** (nearest of
   {primary@θ, Poisson field galaxy@`d_f`}, isolated NaN beyond 3″) — needed at θ>2″;
3. **conditioning weight** `p_det(primary|nbr)/p_det(primary|isolated)` — down-weights bright close
   neighbours that would have destroyed the primary's own clean detection (§5) — needed at θ<1.5″.

All three are essential: primary-only context under-counts (0.82×), naive field context over-counts
(1.31×), conditioned hits **1.03× soft / 1.11× hard**.

| stage | residual Δm |
|---|---|
| drop undetected neighbours | **+3.29%** |
| conditioned forward model (global) | **−0.37%** |
| — population-model geometry | −0.10% |
| — classifier calibration (soft vs hard) | −0.28% |

A **~9× reduction**, near the Stage-IV |m|<0.3% target. The residual is dominated by the
classifier's 8% miscalibration, not the population model.

**Per-primary-magnitude residual has a tilt** that partly cancels in the global: over-predict at
`r_p` 26–27 (+1.1…+1.5%), under-predict at 27–28 (−1.5…−2.3%). The faint-primary end (most
undetected neighbours, classifier least calibrated) is the weak point.

**Classifier recalibration closes the −0.28% (`probblend_calib.py`).** Response-weighted
calibration shows the classifier is *under-confident* in the dominant bin (p_det 0.9–1.0 holds 87%
of the response weight: predicts 0.974, actual detection 0.986) → over-predicts undetectedness.
Isotonic recalibration fit on detection truth drives soft/hard **1.084 → 0.999**, i.e. the
classifier-calibration Δm contribution **+0.28% → −0.00%**. Final global ladder:

| stage | global Δm |
|---|---|
| drop undetected neighbours | **+3.29%** |
| conditioned forward model | **−0.37%** |
| + isotonic classifier recalibration | **≈ −0.10%** (population-geometry floor) |

**Sub-0.3% achieved globally** with no true neighbour positions.

**The per-magnitude tilt is a context-dependent CLASSIFIER miscalibration (`probblend_forward.py`
tilt diagnostic).** The conditioning weight is well-behaved (⟨condw⟩≈0.97 in every bin); the tilt is
that soft/hard *flips sign* with primary magnitude:

| r_p bin | undet hard | undet soft | soft/hard | ⟨p_det_iso⟩ |
|---|---|---|---|---|
| 25–26 | 0.0117 | 0.0138 | 1.18 (over) | 0.96 |
| 26–27 | 0.0186 | 0.0207 | 1.11 (over) | 0.79 |
| **27–28** | **0.0340** | **0.0222** | **0.65 (UNDER)** | **0.10** |

At the faintest primaries (near the detection limit, ⟨p_det_iso⟩=0.10) the truth undetected response
is large (0.034) but the classifier *under*-predicts undetectedness by 35% — opposite sign to the
bright end. The forward model reproduces soft (f/soft≈1.0), so it inherits this.

**Calibration alone does NOT fix the faint bin (`probblend_calib2d.py`, cases 0–7).** Both a global
isotonic and a magnitude-conditional (2D, isotonic within neighbour-magnitude bins) calibration drive
the *global* soft/hard → 1.00, but the r_p 27–28 bin stays at **0.74** either way (worst per-bin tilt
~26%, unchanged):

| r_p bin | hard | raw/h | g-iso/h | 2D/h |
|---|---|---|---|---|
| 18–24 | 1757 | 0.93 | 0.90 | 0.90 |
| 25–26 | 9365 | 1.16 | 1.03 | 1.04 |
| 26–27 | 15928 | 1.12 | 1.04 | 1.03 |
| **27–28** | 2418 | 0.77 | **0.74** | **0.74** |
| GLOBAL | 34246 | 1.09 | 1.00 | 1.00 |

So the faint-primary residual is **not** a calibration artefact fixable by any 1D map of p_det —
it is a **structural limitation of the pairwise classifier**: a neighbour of a *detected faint*
primary sits in a biased, harder-to-deblend local configuration that the single-pair detection model
cannot represent (a multi-object / conditioning effect). Its global impact is small (the faint bin is
~3% of pairs → global m stays ~0.1%), but lensing-weighted per-bin robustness needs a
**richer-context detection model** (full local blend / joint detection), not recalibration.
**Next step for the user.**
**Scripts:** `probblend_characterize.py`, `probblend_forward.py` (conditioned production model),
`probblend_gap_diag.py`, `probblend_ctx_diag.py` (per-θ-shell context resolution).

**Caveats (controlled test):** primary + detected-neighbour props are TRUE (the "measured
properties" idealisation, accepted); measurement noise on them is a separate 2nd-order effect. `Φ`
is the sim's own deep input LF (production: external deep field). m validated on the MAIN set
(has detection truth) via per-magnitude `R_blend` reconstruction; a direct constgold m-test needs a
detection catalogue built on the constant set.

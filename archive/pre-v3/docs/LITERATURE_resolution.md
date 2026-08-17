# Literature: the "resolution problem" — global `m` is fine because per-bin errors cancel

**Scope.** Prior work on the failure mode SBSI is currently in: a shear-calibration model whose
POPULATION-MEAN multiplicative bias is excellent while its PER-BIN bias is large and alternating in
sign, so the global number is right only by cancellation. Compiled 2026-08-02.

**Provenance and trust.** Part 1 (weak lensing) was assembled by a literature agent that fetched and
quoted arXiv abstract pages and paper PDFs directly; the quotations below are its verbatim
extractions, and its own UNVERIFIED list is preserved at the end. **Everything here is second-hand to
this repo — verify any number before it enters a paper or a decision.** Part 2 (statistics / ML) was
requested and is pending; the first attempt died on a session limit.

Why it matters here: SBSI's fiducial constgold `m = -0.123 +- 0.152%` sits against per-bin residuals
of `-4.12%` to `+5.65%` in true size (2026-08-01s), and the tuned emulator is right to **0.8%**
globally while being **3.1x too flat** across size bins. That is this failure mode exactly.

---

## 1. Weak lensing

### 1.1 The strongest formal argument that `m` is not a number

**MacCrann et al. (2022), "DES Y3 results: Blending shear and redshift biases in image simulations",
arXiv:2012.08567, MNRAS 509, 3371.**

Two strands.

*Empirical* — `m` varies by ~3x across DES Y3 tomographic bins (their Table 3, `m x 100`):

| bin 0 | bin 1 | bin 2 | bin 3 |
|---|---|---|---|
| -1.25 +- 0.31 | -1.82 +- 0.39 | -2.27 +- 0.44 | -3.60 +- 0.59 |

*Formal* — under blending the response is a FUNCTION of redshift, not a scalar. Verbatim:

> "In the presence of blending, we can thus no longer assume a separable effective redshift
> distribution n_gamma(z) = R(z) n(z)."

> "the two are expected to disagree, **not just in their normalization (i.e. as an overall
> multiplicative bias), but also in their shape.**"

> "The normalization of n_gamma(z) corresponds to the traditional mean multiplicative bias, 1 + m."

They define `n_gamma(z) = d g_obs / d g_true(z)`, a functional derivative. **A single global `m` is
exactly the normalisation of that function; every bit of its SHAPE is discarded.** This is the
cleanest available citation for "the global number cannot be the deliverable".

They also isolate detection from blending (Table 3): fiducial (random positions + SExtractor)
`m = -2.08 +- 0.12 %`; on a grid `-0.34 +- 0.11 %`; on a grid with TRUE positions (no SExtractor)
`-0.44 +- 0.05 %`. Removing SExtractor from an isolated-object grid changes nothing — the ~2% appears
only with realistic crowding.

### 1.2 Redshift-agnostic calibration is measurably wrong — the closest methodological analogue

**Kannawadi et al. (2019), "Towards emulating cosmic shear data: Revisiting the calibration of the
shear measurements for the Kilo-Degree Survey", arXiv:1812.03983, A&A 624, A92.**

Verbatim abstract:

> "the calibration has to be performed by selecting the tomographic samples in the simulations,
> consistent with the actual cosmic shear analysis, **because the joint distributions of galaxy
> properties are found to vary with redshift. Ignoring this redshift variation could result in
> misestimating the shear bias by an amount that exceeds the allowed tolerance.**"

KV-450 residual `m`, with-redshift vs redshift-agnostic calibration (their Table 2):

| bin | z_B | with-z | no-z | gap |
|---|---|---|---|---|
| B1 | 0.1-0.3 | -0.013 +- 0.008 | **-0.036 +- 0.003** | **0.023** |
| B2 | 0.3-0.5 | -0.010 +- 0.006 | -0.028 +- 0.003 | 0.018 |
| B3 | 0.5-0.7 | -0.011 +- 0.006 | -0.008 +- 0.003 | 0.003 |
| B4 | 0.7-0.9 | +0.007 +- 0.006 | +0.006 +- 0.003 | 0.001 |
| B5 | 0.9-1.2 | +0.006 +- 0.007 | +0.009 +- 0.003 | 0.003 |

The B1 gap (0.023) exceeds their entire 0.02/bin error budget. Note the structure: the gap is
concentrated in one bin and near-zero in three — reweighting a redshift-agnostic simulation does not
recover it.

### 1.3 A zero-MEAN `m` still biases cosmology — the direct answer to "does the global number protect me?"

**Cragg, Duncan, Miller & Alonso (2022), "Propagating spatially-varying multiplicative shear bias to
cosmological parameter estimation for stage-IV weak-lensing surveys", arXiv:2203.01460.**

For a Euclid-like survey with **zero-mean**, spatially varying `m`: rms 0.01 gives parameter biases up
to ~10% of the statistical error at `l_max = 5000`; rms 0.02-0.03 generally exceeds ~30%; rms
0.04-0.05 can exceed the statistical error outright. Their conclusion, verbatim:

> "**requirements should be placed on the rms of spatial variations of the m-bias, in addition to any
> requirement on the mean value.**"

This is the phenomenon in its purest form: mean zero, structure non-zero, cosmology still wrong.

**Samuroff et al. (2018), "DES Y1: The Impact of Galaxy Neighbours on Weak Lensing Cosmology with
im3shape", arXiv:1708.01534.** The cost is not removed by marginalisation. Verbatim:

> "**Even marginalising over m with a prior of N(0, 0.035)** this scenario was demonstrated to result
> in a shift in the favoured cosmology towards low clustering amplitude of **more than 1 sigma**."

The scenario is a per-bin `m` pattern of the kind a neighbour-free calibration simulation would leave
behind. Abstract: neighbours contribute `m ~ 0.03-0.09` in DES Y1 im3shape; reducible to
percent-level by cutting close neighbours at a 30% cost in `n_eff`; omitting blending from the
calibration simulation biases `S_8` by 2 sigma low. Four named mechanisms: direct contamination,
selection bias, S/N bin shifting, neighbour dilution.

### 1.4 Detection bias as a separate axis — the citation chain

- **Kaiser (2000)**, arXiv:astro-ph/9904003 — first notes the effect, but PSF-driven, not
  shear-driven, and does not name it. Buried in the noise-bias section: "there is then a tendency for
  the galaxies close to the threshold for detection to be aligned like the PSF." Scales as `nu^-2`.
- **Bernstein & Jarvis (2002)**, arXiv:astro-ph/0107431, §8.1 — names it "selection bias" and credits
  Kaiser. Mitigation: cut on a shape-independent significance measured on a PSF-rounded image.
- **Hirata & Seljak (2003)**, arXiv:astro-ph/0301054, §3 — **first to establish that detection depends
  on the SHEAR itself**, and coins "shear selection bias". Verbatim: "Unlike PSF-related biases, shear
  selection bias is perfectly (anti-)correlated with the shear signal. This means that shear selection
  bias takes the form of a calibration error, with the fractional error depending on the object
  detection algorithm and on the underlying population of galaxies." Already flags property- and
  redshift-dependence.
- **Sheldon & Huff (2017)**, arXiv:1702.02601 — metacalibration corrects POST-detection selection
  ("better than a part in a thousand"); detection itself is out of scope.
- **Sheldon, Becker, MacCrann & Jarvis (2020)**, arXiv:1911.02505, ApJ 902, 138 — modern name and the
  clean numerical isolation. **The decisive controlled test: same measurement, only the detection
  catalogue changed — `-0.0011 +- 0.0012` with true detections vs `-0.058 +- 0.001` with SExtractor
  detections.** Verbatim: "this bias is not due to blending itself, but rather to shear-dependent
  object detection."

**DO NOT CITE** Bernstein (2010), arXiv:1001.2333, for detection bias — verified to be about
underfitting and ellipticity-gradient bias, with no detection content.

### 1.5 Two findings that bear directly on SBSI's conventions

**(a) Leg matching can re-inject the bias.** Sheldon et al. (2020) §4.3, verbatim, on why per-object
responses are unusable:

> "Doing so would require matching the lists of detections found on the different sheared images...
> This act of matching would introduce the very shear-dependent object detection biases we wish to
> calibrate."

SBSI's both-detected leg matching is exactly this operation. `CONVENTIONS.md` already records what
both-detected silently removes; this is independent published support for treating it as a bias
source, not bookkeeping.

**(b) Cut on TRUE properties, not measured.** Euclid Collaboration: Congedo et al. (2024),
arXiv:2405.00669, A&A 691, A319, verbatim: "defining true input bins is also essential to minimise the
impact of selection bias and not to misinterpret results." Matches SBSI's true-vs-measured cut rule —
and is corroborated by this repo's own 2026-08-01p/r finding that a MEASURED magnitude cut is
substantially a blending cut.

### 1.6 Euclid: detection bias dominating the error budget

**Congedo et al. (2024)**, arXiv:2405.00669. Biases averaged over `20 < I_E < 24.5` (their Table 2):

| | measurement | selection |
|---|---|---|
| `m_1 / 1e-3` | -3.58 +- 0.18 | **11.8 +- 1.0** |
| `m_2 / 1e-3` | -4.30 +- 0.18 | **0.0 +- 1.0** |

"Selection" covers detection, catalogue matching, star-galaxy separation and weights. The
`m_1` / `m_2` asymmetry (+1.18% vs exactly 0.00%) is real and flagged by the authors as unexplained
after ruling out star-galaxy separation, weights and the particular PSF. Undetected faint objects
alone contribute `m_faint ~ -5e-3`; they recommend rendering neighbours to magnitude >= 27.5.

Supporting:
- **Euclid Collaboration: Martinet et al. (2019)**, arXiv:1902.00044, A&A 627, A59 — undetected
  galaxies give `m ~ few x 1e-3` when uniformly distributed, rising to **`~1e-2` once their
  clustering is included**. Neighbours must be simulated down to magnitude 28.
- **Euclid Collaboration: Csizi et al. (2025)**, arXiv:2409.07528, A&A 695, A283 — "we find clear
  detection bias differences between full image scenes simulated with parametric and realistic
  galaxies, leading to a bias difference of `4.0e-3` **independent of the shape measurement
  method**." The strongest statement that detection is a separate axis from shape measurement.
- **Euclid Collaboration: Jansen, Martinet et al. (2026)**, arXiv:2604.26684 — refining input shape
  distributions changes `m` at the percent level, "exceeds Euclid's tight error budget by a factor of
  five". No journal ref yet.
- **Voigt (2025)**, arXiv:2512.00666 — faint blends give `m = -0.008`; biases depend not only on mean
  faint-galaxy density but on **how it varies with the bright galaxy's own brightness**, and on radial
  / tangential alignment and positional anisotropy. Shear coherence, parallel orientation alignment
  and size-magnitude variation are negligible.

### 1.7 Property-dependence, other surveys

- **Mandelbaum et al. (2018)**, arXiv:1710.00885 (HSC Y1) — removes "galaxy property-dependent
  multiplicative and additive shear biases", including a **~10 per cent-level** multiplicative bias
  from nearby galaxies and unrecognised blends; final catalogue controlled at the 1% level.
- **Fenech Conti et al. (2017)**, arXiv:1606.05337 (KiDS) — calibration "also corrects for a
  dependence of the bias on galaxy properties", and warns "the calibration relation itself is biased
  by the use of noisy, measured galaxy properties".
- **Hoekstra, Viola & Herbonnet (2017)**, arXiv:1609.03281 — algorithms must account for the local
  density of sources; "magnification changes their number density, resulting in correlations between
  the lensing signal and multiplicative bias".
- **Yamamoto, Becker, Sheldon et al. (2025)**, arXiv:2501.05665 — DES Y6, first on-sky metadetection,
  `m = (3.4 +- 6.1) x 1e-3`.
- **Melchior et al. (2018)**, arXiv:1802.10157 (SCARLET) — a photometry/morphology deblender; its
  abstract makes NO claim about shear or detection bias, and no verified paper measures a shear
  detection bias for SCARLET specifically. Sheldon et al. (2020) argues deblending QUALITY is largely
  beside the point.

### 1.8 What the field does about it

*Diagnose*: report `m` as a function of S/N and size (the classic `m(S/N, R)` surface); per-tomographic-bin
`m` with its own error; `m` vs true pair separation (Sheldon et al. 2020 Fig. 4, MacCrann et al. 2022
Fig. 7); `m` vs true input magnitude (Congedo et al. 2024 Fig. 10).

*Mitigate*: calibrate WITHIN the tomographic selection rather than reweighting a redshift-agnostic
simulation (Kannawadi et al. 2019); cut close neighbours, at a real `n_eff` cost (Samuroff et al.
2018: 30%); put detection INSIDE the calibration loop (metadetection, Sheldon et al. 2020 — reduces
`~ -3.5%` to `< 1e-3`); cut on a shape-independent significance (Bernstein & Jarvis 2002); simulate
faint neighbours deep enough and with correct clustering (Martinet et al. 2019).

*The honest limitation*: MacCrann et al. (2022) state they "cannot fully decouple" detection bias from
neighbour contamination — the decomposition SBSI is attempting is not one the field has cleanly
achieved either.

### 1.9 Agent's UNVERIFIED list — do not cite these without checking

- The Melchior et al. blending review in Nature Reviews Physics — could not be located; arXiv:2107.05846
  is an unrelated quantum-networks paper.
- The Mau, Becker et al. DES Y6 image-simulation paper referenced by Yamamoto et al. (2025).
- Numeric details inside Sheldon et al. (2023), arXiv:2303.03947 (Rubin/LSST metadetection) — abstract
  only; PDF not opened.
- Published volume/page for Fenech Conti et al. (2017) and Hoekstra et al. (2017).
- Whether the promised follow-up Euclid selection/detection-bias paper exists.
- MacCrann et al. (2022) Table 3 bin-3 per-component `m_1`/`m_2` (PDF extraction garbled); the
  combined `-3.60 +- 0.59` is verified.

---

## 2. Statistics / machine learning

Same provenance caveat: assembled by an agent that read the arXiv PDFs directly; its UNVERIFIED list
is §2.7. Where it verified only the bibliographic record and not the text, that is flagged.

### 2.1 The hierarchy — the single most useful framing

For a mean-response model `S = m(X)` predicting `Y`, with `Q(X) = E[Y|X]` the truth:

1. `E[S] = E[Y]` — global mean match. **This, and only this, is what a global `m ~ 0` tests.**
2. `E[Y | S=s] = s` for all `s` — calibration. Strictly stronger.
3. `S = Q(X)` almost surely — no grouping loss. Strictly stronger again.

**And the part that matters most for SBSI:** a selection cut on a variable `Z` reports
`E[Y - S | Z in A]`. That is non-zero exactly when the residual `Y - S` correlates with the indicator
`1{Z in A}` — which is the **multiaccuracy** condition, *not* full calibration.

> **Practical consequence:** if the cuts we care about are known indicator sets (`mag < 25`,
> `Re > 0.6`), the property we need is multiaccuracy over the class of CUT SETS. That is much weaker
> and cheaper than multicalibration. Multicalibration is only needed additionally if we also cut on
> the model's own output.

### 2.1b The four-term decomposition — SBSI's situation, named exactly

Kull & Flach's Thm. 4 splits the calibration term further, with `A = alpha(S)` an *adjusted* score
whose mean is matched to the base rate:

```
Loss  =  AL          +  PCL              +  GL         +  IL
         adjustment     post-adj. calib.    grouping     irreducible
```

**Adjustment fixes only the global mean. Calibration fixes the whole score->probability map. Neither
touches GL.** A model with the population mean right and the property-dependence wrong is precisely
one with `AL ~ 0` and `GL` large. That is the fiducial constgold model in one line.

### 2.2 Murphy (1973) — "resolution" is literally the term, and the identity is EXACT

**Allan H. Murphy, "A New Vector Partition of the Probability Score", J. Appl. Meteorol. 12(4),
595-600, 1973**, DOI `10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2`.

With `S = f(X)` and `C(S) = E[Y|S]`:

```
E[(S-Y)^2]  =  E[(S-C)^2]   -   Var(C)      +   Var(Y)
               RELIABILITY      RESOLUTION      UNCERTAINTY
               (calibration)    (good, so -)    (irreducible)
```

Resolution enters with a MINUS sign — it is a virtue, and it is exactly the variance of the outcome
rate across the classes the forecast separates. **A model emitting the population mean for everything
is perfectly reliable and has ZERO resolution.** So "resolution problem" is not loose physicist
shorthand; it is the standard forecast-verification term for this deficiency.

**THE IDENTITY IS EXACT, not an analogy.** Combining Kull & Flach's `RL = UNC - RES` (their §7) with
`RL = GL + IL` (their Thm. 2), and `UNC - IL = Var(Q)` for binary Brier:

```
        RES = Var(Q) - GL        <=>        GL = Var(Q) - Var(C)
```

which is the law of total variance, `Var(Q) = Var(E[Q|S]) + E[Var(Q|S)]`. **The maximum achievable
resolution is `Var(Q)`, the total signal available, and the grouping loss is exactly the part of it
the model failed to resolve.** The agent verified these identities numerically (Monte Carlo,
`N = 4e6`; machine precision on the exact ones). So "resolution problem" and "grouping loss" name the
same quantity.

**Where Murphy alone stops.** He conditions on the FORECAST VALUE, not on covariates. Zero resolution
means a *constant* model. A model that varies with `x` but varies **wrongly** has non-zero resolution
and can still be perfectly reliable — hence the need for §2.3.

**Two terminology traps.**
1. Murphy (**1972**) used "resolution" for what is now called *refinement loss*; Murphy (**1973**)
   reused the same name for a different quantity, `Var(C)`. Kull & Flach flag this explicitly. Always
   check which is meant.
2. *sharpness* (Bross 1954 / Sanders 1963) = *resolution* (Murphy & Winkler 1977) = *refinement*
   (DeGroot & Fienberg 1983) = *sufficiency* (DeGroot & Fienberg) all name one concept across four
   literatures.

Also: **DeGroot & Fienberg (1983)**, "The Comparison and Evaluation of Forecasters", The Statistician
32(1/2), 12-22 — generalised calibration+refinement to all proper scoring rules and stated the
normative rule *minimise refinement loss subject to calibration*. **Broecker (2009)**, QJRMS 135(643),
1512-1519 — generalised Murphy's three-term partition to arbitrary proper scores; the reference for
doing this outside Brier.

### 2.3 Grouping loss — the exact formal home

**Kull & Flach (2015)**, "Novel Decompositions of Proper Scoring Rules for Classification", ECML PKDD,
DOI `10.1007/978-3-319-23528-8_5`. For any strictly proper scoring rule with divergence `d`, with
`Q = P(Y|X)` the true posterior, `S = f(X)` the score, `C = E[Q|S]` the calibrated score:

```
E[d(S,Y)]  =  E[d(S,C)]   +   E[d(C,Q)]     +   E[d(Q,Y)]
              CALIBRATION     GROUPING loss     IRREDUCIBLE
```

**Grouping loss is the error that survives after conditioning on the model's own output** — samples
sharing one score but having different true values. Global correctness kills neither term.

**Perez-Lebel, Le Morvan & Varoquaux (2023)**, "Beyond calibration: estimating the grouping loss of
modern neural networks", ICLR 2023, arXiv:2210.16315. Their **Lemma 4.1** rewrites GL as a variance —
for the Brier case `GL(S) = E[ E[ ||Q - C||^2 | S ] ]`. **Grouping loss IS literally the variance of
the truth inside a level set of the model output**, i.e. the alternating-sign subgroup errors that
cancel. That is our phenomenon, exactly.

What makes this paper directly usable:

- **`Q(X)` is unknown so GL is not computable — they give an estimable LOWER BOUND.** Theorem 4.1
  splits `GL = GL_explained + GL_residual` over a partition of feature space, with `GL >= GL_explained
  >= 0`, and `GL_explained` needs only an empirical mean of `Y` per region.
- **Binning inflates GL** (Prop. 4.1): `GL(S_B) = GL(S) + GL_induced >= GL(S)`. The ECE-style binning
  that makes estimation feasible biases the answer UPWARDS. Their usable bound (Prop. 4.2) subtracts
  it: `GL >= GL_explained(S_B) - GL_induced(S, S_B)`, both terms estimable.
- **A debiased Brier estimator** (Prop. 4.3): plug-in minus an explicit finite-sample variance
  correction. The plug-in alone "substantially overestimates" GL.
- **Practical sizing:** ~15 bins, regions targeting ~a dozen samples each; below ~2 samples/region the
  estimator breaks. Partition from a decision tree fitted on `Y` with the scoring-rule loss on one
  half, region means on the held-out half.
- **THE FINDING THAT KILLS THE OBVIOUS FIX, and it is a THEOREM not an observation.** Their
  **Lemma C.5**: for any recalibration map with `S' = c(S)`,
  ```
  GL(S') = GL(S) + E[ V_h[C | S'] ]  >=  GL(S)
  ```
  with equality iff the map is perfect or invertible. **Recalibration NEVER decreases grouping loss,
  and a non-injective map strictly INCREASES it.** Confirmed empirically in the same paper for
  isotonic regression. Platt scaling, temperature scaling and isotonic regression are all monotone
  maps *on the score* — they never look at the inputs, so they cannot repair a wrong dependence on
  inputs. (An earlier version of this file said "leaves it essentially unchanged"; that was too weak.)
  Compounding this: **Kumar, Liang & Ma (2019)**, "Verified Uncertainty Calibration", NeurIPS,
  arXiv:1909.10155 — measured ECE is itself optimistically biased, so the reassurance is doubly false.
- GL is concentrated in **distribution-shift / out-of-distribution** settings.
- They explicitly name **simulation-based inference** as a downstream task harmed by grouping loss.
- Their **"grouping diagram"** is a directly transferable plot: calibration curve overlaid with
  per-region empirical rates and Clopper-Pearson 95% CIs; regions whose CI excludes the calibrated
  score are the flagged subgroups.

### 2.4 Multicalibration — guarantees over groups you did not enumerate

**Hebert-Johnson, Kim, Reingold & Rothblum (2018)**, arXiv:1711.08513 (arXiv title: "Calibration for
the (Computationally-Identifiable) Masses"; the "Multicalibration:" prefix is the ICML title).
`alpha`-multicalibrated w.r.t. a family `C` = `alpha`-calibrated on **every** `S in C`, conditioning
JOINTLY on the subgroup and on the predictor's level set. `C` is any family with efficiently
computable membership; sets may overlap.

- **Thm 2 (sample complexity)** `O(log|C| / alpha^{11/2} gamma^{3/2})`. Algorithm is iterative: find a
  violated `(S,v)` pair, patch, repeat.
- **Thm 4 (lower bound)** multicalibration learning IMPLIES weak agnostic learning — it is *exactly as
  hard as agnostic learning over `C`*, `Omega(|C|^t)` in the worst case.
- **Thm 5 (accuracy is not sacrificed)** there is an `alpha`-multicalibrated predictor within `6 alpha`
  of the best in class. **Enforcing calibration after learning costs little; if it changes predictions
  a lot, that change is an improvement.**

**Kim, Ghorbani & Zou (2018)**, arXiv:1805.12317, MULTIACCURACY-BOOST — the weaker, cheaper condition
`|E[(y - p(x)) c(x)]| <= alpha` for all `c in C`, no level-set conditioning. Black-box access plus a
small labelled audit set; an auditor regresses the residual on features and the fitted direction is
subtracted. **This is the cheap target identified in §2.1.**

**Regression extensions — what makes this usable for a continuous forward model:**
- **Jung, Lee, Pai, Roth & Vohra (2021)**, "Moment Multicalibration for Uncertainty Estimation", COLT,
  arXiv:2008.08037 — multicalibrates means AND variances simultaneously across many overlapping
  subgroups, giving per-subgroup-valid prediction intervals.
- **Globus-Harris, Harrison, Kearns, Roth & Sorrell (2023)**, "Multicalibration as Boosting for
  Regression", ICML, arXiv:2301.13767 — characterises multicalibration via a swap-regret condition on
  squared error; **needs only a standard squared-error regression oracle**, converges to Bayes
  optimality with no realizability assumption. This is the practical entry point.
- Also: Gopalan et al., "Omnipredictors" (arXiv:2109.05389) and "Low-Degree Multicalibration" (COLT
  2022, arXiv:2203.01255, trades strength for tractability); Shabat, Cohen & Mansour, "Sample
  Complexity of Uniform Convergence for Multicalibration" (NeurIPS 2020, arXiv:2005.01757).

**The sceptical companion, read before investing:** **Hansen, Devic, Nakkiran & Sharan (2024)**, "When
is Multicalibration Post-Processing Necessary?", NeurIPS, arXiv:2406.06487 — for well-trained models,
ordinary ERM plus simple recalibration is often already close to multicalibrated, and the
post-processing buys less than advertised.

**No verified application of multicalibration in physics or astronomy was found** (§2.8 gap 2).

### 2.5 The impossibility result that bounds any fix

**Barber, Candes, Ramdas & Tibshirani (2021)**, "The limits of distribution-free conditional
predictive inference", Information and Inference 10(2), 455-482, arXiv:1903.04684.

- **Prop. 1** (after Vovk 2012, Lei & Wasserman 2014): any method with exact conditional coverage has
  **infinite expected interval length** at almost all non-atomic points. Exact conditional coverage is
  unattainable distribution-free.
- **Thm 2**: even the relaxation "coverage on every set of probability >= delta" forces intervals
  essentially no better than the trivial construction.
- **Thm 3 — what IS achievable**: coverage over a *declared class* `X` of subgroups is attainable, via
  a locally-widened quantile. **Thms 4/5**: the width cost scales with `VC(X)`. **The richer the
  subgroup class you demand, the wider the intervals — complexity buys width.**

**Vovk (2012)**, arXiv:1209.2673, §4 gives the practical mechanism under the name *conditional
inductive conformal prediction*: an inductive taxonomy assigns each example a category, p-values are
computed WITHIN category, and **Prop. 3** guarantees the error probability conditional on the category.
*Define categories = the physics subgroups, calibrate inside each.* (Cite Vovk 2012, not the 2003
"Mondrian" report — see §2.7.)

**Gneiting, Balabdaoui & Raftery (2007)**, JRSS-B 69(2), 243-268 — the framing "maximise sharpness
subject to calibration". In this vocabulary, "right on average but uninformative" is a SHARPNESS
deficit; "wrong within subgroups" is a CALIBRATION deficit.

### 2.6 SBI diagnostics — global tests will pass while we are wrong

| test | reference | global or local |
|---|---|---|
| SBC | Talts et al., arXiv:1804.06788 | **GLOBAL** — ranks averaged over the prior; opposite-sign subgroup errors cancel in the rank histogram |
| expected coverage | Hermans et al., arXiv:2110.06581 (TMLR; title "A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful") | **GLOBAL** — averaged over the prior |
| **LCT / ALP** | Zhao, Dalmasso, Izbicki & Lee, arXiv:2102.10473, UAI 2021 | **LOCAL** — identifies *where* in feature space the conditional density departs from truth |
| TARP | Lemos et al., arXiv:2302.03026, ICML 2023 | coverage without density evaluations; conditions necessary AND sufficient |
| **L-C2ST** | Linhart, Gramfort & Rodrigues, arXiv:2306.03580, NeurIPS 2023 | **LOCAL** — evaluates the estimator at a given observation, no true-posterior samples; more powerful than HPD coverage for normalizing flows |

Two additions:
- **Cal-PIT / LADaR** — Dey, Zhao, Andrews, Newman & Izbicki, MLST 2025, arXiv:2205.14568. Local PIT
  calibration **applied in astronomy** (photometric redshifts). The nearest precedent in our own field.
- **LF2I** — Dalmasso, Masserano, Zhao, Izbicki & Lee, EJS 18(2), 2024, arXiv:2107.03920.

**SBC's blind spot is documented, not folklore:** **Modrak, Moon, Kim, Buerkner & Huurre (2023)**,
"Simulation-Based Calibration Checking for Bayesian Computation: The Choice of Test Quantities Shapes
Sensitivity", Bayesian Analysis, arXiv:2211.02383 — SBC's sensitivity depends entirely on which test
quantity you choose, and default choices miss whole classes of error. **If SBC is the acceptance test,
read this first.**

**Recommendation: SBC and expected coverage are exactly the tests that pass while the model is wrong
within subgroups.** Use LCT/ALP or L-C2ST, read as functions of the physical property (magnitude,
size, blend state) the model is suspected to mis-track. Zhao et al.'s construction is the most
directly applicable: regress the PIT value on the conditioning variables (amortised, no binning), then
test whether the fitted surface is flat. **A non-flat surface IS the resolution failure, displayed as a
function of the property that drives it.**

### 2.7 Remedies, and what each CANNOT fix

| remedy | fixes | cannot fix |
|---|---|---|
| group-wise recalibration | mean and calibration curve on groups you NAME | groups you did not name; and **post-hoc recalibration leaves grouping loss essentially unchanged** (Perez-Lebel Figs. 6/7) — a monotone map of the score cannot separate points sharing a score |
| importance weighting to a target | makes the aggregate right under THAT target | **does not change the pointwise error `S(x) - Q(x)` at all — it relocates the cancellation point.** Right for one target, wrong for another, wrong under any cut. Weight variance explodes under weak overlap |
| group DRO (Sagawa et al., arXiv:1911.08731) | worst-group loss; 10-40 pt worst-group gains. Key finding: **strong regularisation is essential** for worst-group generalisation even when unnecessary for average | needs group labels AT TRAINING TIME; protects only annotated groups; real worst-group/average trade-off |
| multiaccuracy / multicalibration boosting | guarantees over a whole CLASS including groups never enumerated; accuracy cost `< 6 alpha` | only sets in `C`; an `alpha` floor; cost of agnostic learning; needs held-out labels |
| adding interaction / conditioning features | attacks GL directly — moves mass from GL into IL **only if the feature is genuinely informative** | if the missing dependence is on an unmeasured variable, GL is irreducible w.r.t. your feature set — you have relabelled it, not fixed it. Vulnerable to errors-in-variables in the added feature |
| group-conditional conformal | finite-sample coverage inside each declared category | coverage, not point accuracy; categories declared in advance, each needs data; width grows with `VC(X)` |
| **Sobolev training — supervise the DERIVATIVE directly** (Czarnecki, Osindero, Jaderberg, Swirszcz & Pascanu 2017, NeurIPS, arXiv:1706.04859) | **precisely the "value right, slope wrong" pathology.** If the target IS a response (a derivative w.r.t. a physical parameter), training on values alone does not constrain it — the loss is nearly flat along directions that change the derivative while preserving the marginal fit | needs derivative targets (**here obtainable from finite differences between shear legs**); ~doubles supervision per sample. **No published application to conditional normalizing flows** — this would be a genuine extension, not off-the-shelf. Follow-ups extend Sobolev training to PINNs (arXiv:2101.08932) and operator learning (arXiv:2402.09084) |

**Two more sceptical companions on the remedies above.** **Byrd & Lipton (2019)**, "What is the Effect
of Importance Weighting in Deep Learning?", ICML, arXiv:1812.03372 — for over-parameterised networks
trained to convergence, the effect of importance weighting **diminishes over training and can vanish
entirely**. **Gulrajani & Lopez-Paz (2021)**, arXiv:2007.01434 (tuned ERM matches most robustness
methods) and **Idrissi et al. (2022)**, arXiv:2110.14503 (simple subsampling matches group DRO). The
negative results in this area are as well-established as the positive ones.

**The hard bound on all of it.** Barber et al.'s Theorem 2 means **no method, existing or future, can
deliver distribution-free guarantees at the level of individual inputs or arbitrary subgroups without
either infinite-width intervals or being no better than trivial marginal coverage.** Every remedy above
works by restricting attention to a DECLARED class of subgroups. There is no version where the
guarantee comes free across all of property space; a claim otherwise is either assuming smoothness
(fine, but say so) or wrong.

### 2.7b Conditional-model failures closest to ours

- **Wiese, Knobloch & Korn (2019)**, "Copula & Marginal Flows: Disentangling the Marginal from its
  Joint", arXiv:1907.03361 — argues standard flows conflate marginal behaviour with dependence
  structure and cannot exactly control either. **The only work found that directly targets "marginals
  right, dependence wrong" in flows** — which is this repo's 2026-08-01r diagnosis.
- **Cremer, Li & Duvenaud (2018)**, "Inference Suboptimality in VAEs", ICML, arXiv:1801.03558 — splits
  the gap into an approximation gap and an **amortization gap**: error that varies across inputs while
  the average looks acceptable. Conceptually the closest existing framing to "the conditional response
  is wrong but the mean is right" in a learned conditional model.
- **Theis, van den Oord & Bethge (2016)**, ICLR, arXiv:1511.01844 — log-likelihood, sample quality and
  other criteria are largely independent. The general licence for "your global metric is not measuring
  what you think".

### 2.8 Agent's UNVERIFIED list (part 2)

Bibliographic records verified but **full text not read**: Murphy (1973) — the three-term formula and
signs are transcribed from standard knowledge; Kull & Flach (2015) — content attributed via
Perez-Lebel et al.'s explicit citation at their Eq. 6; Gneiting et al. (2007) — summary from standard
knowledge. **"Mondrian conformal predictors"** as a named 2003 reference is NOT verified — the
mechanism IS verified as conditional ICP in Vovk 2012 §4 Prop. 3, so cite that. Venue/page records not
verified for: Hebert-Johnson et al. (ICML 2018 PMLR), Kim et al. (AIES 2019), Sagawa et al. (ICLR
2020), Talts et al. (journal), Lemos et al. and Linhart et al. (proceedings pages). **Shimodaira
(2000)** for covariate-shift importance weighting — not checked at all. **No application of
multicalibration in physics or astronomy was found** — the agent explicitly does not assert one
exists.

---

### 2.9 GAPS — searched for and NOT found

Stated as absence-of-evidence, which for a well-indexed literature is reasonably strong. The agent
used arXiv full-text search plus targeted queries.

1. **Grouping loss is defined only for CLASSIFICATION.** Every source (Kull & Flach 2015; Perez-Lebel
   2023; Tasche 2021; Chen 2024) is built on `Q = P(Y=1|X)` with categorical `Y`. **For a conditional
   density model with continuous outputs there is no off-the-shelf grouping-loss formalism or
   estimator.** The `h`-variance formulation looks like it should generalise via the Jensen gap of any
   strictly proper scoring rule for continuous outcomes (CRPS, energy score), but nobody appears to
   have done it. **A real and probably tractable gap.**
2. **Multicalibration has essentially zero penetration into the physical sciences.** arXiv search for
   multicalibration co-occurring with physics/astronomy/climate returned nothing. The regression
   extensions exist and are directly applicable, but nobody has applied them to a physics emulator.
3. **No documented "conditioning collapse" for conditional normalizing flows.** Posterior collapse is
   well-established for VAEs (Bowman 2016 arXiv:1511.06349; Alemi 2018 arXiv:1711.00464; He 2019
   arXiv:1901.05534) and conditioning collapse for class-conditional GANs (Shahbazi et al. 2022,
   ICLR, arXiv:2201.06578). The flow analogue — a conditional flow whose context effect degenerates
   into a mean shift — returned nothing. Since flows train by exact likelihood rather than an ELBO
   with a KL term, the mechanism would differ, so this **may be genuinely unstudied**. *This is
   exactly this repo's 2026-08-01r diagnosis.*
4. **Nothing connects the grouping-loss / multicalibration formalism to RESPONSE or DERIVATIVE
   quantities.** All of this literature evaluates predicted *values*; our target is a *derivative*
   w.r.t. a physical parameter. "The model's response is right on average but wrong per subgroup" is
   one level up from anything in the literature, and **a grouping loss on the response rather than on
   the prediction does not appear to be defined anywhere.**
5. **Nearest physics precedents, both thin:** Dey et al. (2025), MLST, arXiv:2205.14568 (local PIT for
   photometric redshifts) and Mendes et al. (2026), arXiv:2602.19363 (Mondrian group-conditional
   conformal for a neutron-star EoS surrogate). Neither uses grouping loss or multicalibration; both
   are conditional-coverage rather than resolution framings.

---

## 3. What this implies for SBSI

Stated as implications, not results — none of this has been run here.

1. **Global `m` tests only rung 1 of the §2.1 hierarchy.** The fiducial `-0.123 +- 0.152%` is a
   statement about `E[S] = E[Y]` and nothing more. Weak lensing (MacCrann 2022) and statistics
   (Kull & Flach) reach that independently. In the four-term language (§2.1b) we are `AL ~ 0` with
   `GL` large.
2. **The cheap target is multiaccuracy over the CUT SETS, not full multicalibration.** Our cuts are
   known indicator sets, so we need the residual uncorrelated with those indicators — far weaker and
   cheaper than calibrating on every level set. Entry point for the continuous case:
   Globus-Harris et al. 2023 (needs only a squared-error regression oracle).
3. **A per-bin recalibration provably CANNOT fix it** (Lemma C.5) and a non-injective map makes it
   worse. Independently supports AGENTS.md's "no silent empirical corrections" rule — for a technical
   reason, not a governance one.
4. **Perez-Lebel's estimator would give a NUMBER** for the flow's grouping loss, and their grouping
   diagram is directly transferable to `m` vs true size / magnitude. Caveat: it is a LOWER bound, so it
   can convict but never acquit, and its tightness depends on choosing a partition aligned with the
   property that drives the variation — an advantage here, since our suspect properties are known
   physically.
5. **Our SBI-side validation should be LOCAL.** SBC and expected coverage are precisely the tests that
   pass while per-bin structure is wrong; Modrak et al. 2023 documents SBC's blind spot directly.
6. **Adding conditioning features only helps if the missing dependence is on something we measure.**
   The 2026-08-01r finding is a missing dependence on the measurement REALISATION, not on a feature —
   so more features will not reach it.
7. **The best-matched remedy in the whole review is Sobolev training** (§2.7): supervise the
   derivative, not just the value. Our target IS a derivative, our loss trains on values, and the loss
   is nearly flat along directions that change the response while preserving the marginal fit. **We
   already have derivative targets — the finite difference between shear legs.** Note this would be a
   genuine extension: no published application to conditional normalizing flows.
8. **Two of the literature's open gaps are exactly our problem** (§2.9 items 3 and 4): conditioning
   collapse in flows, and grouping loss defined on a response rather than a prediction. If the work
   here is written up, that is where the novelty sits.

# INFERENCE.md — inferring shear, its response, and selection bias from the forward model

**Status:** derivation note, 2026-07-25. Complements `SBI_shear.md` (the framework spec) and
`Gold-v1.md` / `Gold-V2.md` (the certified numbers). This document supplies the *estimator algebra*
that `SBI_shear.md:192-196` gestures at ("the response is internal to it… just the curvature
$\mathcal L''(0)$") but never writes down, and maps it onto our actual checkpoints and scripts.

**One-line summary.** The shear response is a **covariance with the score**, and selection bias is
the **size/flux channel of that same score**, made visible by the cut. Both follow from one identity;
neither requires a new model.

**Three results recorded here that are not elsewhere in the repo:**

1. The response can be obtained from data without paired-sim differencing — via Fisher's identity,
   the covariance (likelihood-ratio) identity, and Bartlett/Louis (§2).
2. `R_blend` enters a Bayesian inference as the **neighbour channel of the score**, combined by
   covariance rather than summed by hand — and it is **identically zero** unless the likelihood
   depends on neighbour *geometry* (§3). This is why `Gold-v1.md` must add BlendEMU externally.
3. The **posterior mean cannot calibrate selection bias** — structurally, not just practically — and
   the boundary formula that replaces it factorizes cont.156's `size>4.4` gap into three measurable
   pieces, two of which are already verified, proving a *mean-only* response pin can never close it
   (§4, §5).

---

## 1. Setup and notation

| symbol | meaning |
|---|---|
| $\mathbf{x}$ | true properties of the primary: shape $e$, size $T$, flux, Sérsic $n$ |
| $\mathbf{n}$ | neighbour configuration (separations, position angles, shapes, fluxes) |
| $\hat{\mathbf{x}}$ | measured catalogue quantities; for the V2/S2 flow $\hat{\mathbf{x}} = (\hat e_1, \hat e_2, \hat m, \hat s)$ |
| $p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ | the learned flow — **shear-free** |
| $p_0$ | intrinsic (unsheared) true-property prior |
| $S_\gamma$ | analytic shear map acting on **truth** |
| $W(\hat{\mathbf{x}})$ | analyst's cut indicator, e.g. $\mathbb 1[\hat s > s_c]$ |

Shear appears in exactly one place — it shifts the prior:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,
p_0\big(S_{-\gamma}(\mathbf{x},\mathbf{n})\big)\,d\mathbf{x}\,d\mathbf{n}.$$

Everything below is a consequence of this single structure.

---

## 2. The score, and three identities

Define the **score**

$$s(\hat{\mathbf{x}}) \;\equiv\; \frac{\partial}{\partial\gamma}\log p(\hat{\mathbf{x}}\mid\gamma)\Big|_{\gamma=0}.$$

### 2.1 Fisher's identity — the score of the marginal is a posterior average

$$\partial_\gamma p(\hat{\mathbf{x}}\mid\gamma)=\int p(\hat{\mathbf{x}}\mid\mathbf{x})\,p_\gamma(\mathbf{x})\,
\partial_\gamma\log p_\gamma(\mathbf{x})\,d\mathbf{x}$$

Dividing by $p(\hat{\mathbf{x}}\mid\gamma)$, the bracket becomes the posterior:

$$\boxed{\;\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)
=\mathbb E_{p(\mathbf{x}\mid\hat{\mathbf{x}})}\big[\partial_\gamma\log p_\gamma(\mathbf{x})\big]\;}$$

**Why this matters operationally:** $\partial_\gamma\log p_\gamma(\mathbf{x})$ is *analytic* — shear
enters truth through the closed-form map $S_\gamma$. **The flow is never differentiated with respect
to $\gamma$** (it cannot be: $\gamma$ is not one of its inputs). All you need is the posterior over
true properties, which `sbs_shear/posterior_shape.py` already computes on a grid.

The analytic factor is the generator of the transport. With velocity field
$v(\mathbf{x})\equiv\partial_\gamma S_\gamma(\mathbf{x})|_0$, the continuity equation
$\partial_\gamma p_\gamma+\nabla\!\cdot\!(p_\gamma v)=0$ gives

$$u(\mathbf{x})\;\equiv\;\partial_\gamma\log p_\gamma(\mathbf{x})\big|_0
\;=\;-\Big(v\cdot\nabla\log p_0(\mathbf{x})+\nabla\!\cdot\! v\Big).$$

### 2.2 The covariance identity — response without finite differences

For **any** fixed statistic $f(\hat{\mathbf{x}})$ (the function is fixed; only the density moves):

$$\frac{d}{d\gamma}\mathbb E_\gamma[f]
=\int f\,\partial_\gamma p
=\int f\,p\,\partial_\gamma\log p
=\mathbb E_\gamma[f\,s],$$

and since $\mathbb E_0[s]=0$,

$$\boxed{\;\frac{d}{d\gamma}\mathbb E_\gamma[f]\Big|_0=\mathrm{Cov}_0\big(f(\hat{\mathbf{x}}),\,s(\hat{\mathbf{x}})\big)\;}$$

This is the likelihood-ratio / score-function estimator (Glynn 1990; Williams 1992 "REINFORCE").
Setting $f=\hat e$ gives the shear response as a **sample covariance on the observed catalogue** —
no paired $\pm g$ renders, no $R_{\rm sim}$.

### 2.3 Bartlett and Louis — the normalization comes free

$$\mathbb E[s]=0,\qquad \mathcal I\equiv\mathrm{Var}[s]=-\,\mathbb E[\partial_\gamma s],$$

and Louis (1982) for the observed information of a latent-variable model:

$$\partial^2_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)
=\mathbb E_{\rm post}\big[\partial^2_\gamma\log p_\gamma\big]
+\mathrm{Var}_{\rm post}\big[\partial_\gamma\log p_\gamma\big].$$

Hence the estimator

$$\hat\gamma=\frac{\sum_i s_i}{\sum_i \mathcal I_i}
\qquad\Longrightarrow\qquad
\frac{\partial\,\mathbb E[\hat\gamma]}{\partial\gamma}=1\ \text{ by construction.}$$

There is no external responsivity to apply afterwards, and no $R_{\rm self}+R_{\rm blend}$ to add by
hand. By Cramér–Rao it is also the minimum-variance first-order estimator: the per-object weight is
its own Fisher information, rather than something tuned.

### 2.4 Sanity limits

- **Noiseless measurement.** The posterior collapses to a delta, $s\to u(\mathbf{x})$, and
  $\mathrm{Cov}(e,s)=\partial_\gamma\langle e\rangle$ — the Bernstein & Jarvis (2002) responsivity
  $\mathcal R=2(1-e_{\rm rms}^2)$ in the distortion convention, $\mathcal R\to1$ in the
  reduced-shear ($\varepsilon$, ngmix `g`) convention. Our shape velocity is
  $v_e=\partial_g\big[(e+g)/(1+\bar g e)\big]_0 = (1-e^2,\;i(1+e^2))$ for $(g_1,g_2)$, whose isotropic
  average is exactly $1$. ✔ consistent with `SBI_shear.md:265`.
- **No cut, shape only.** Only $s_{\rm shape}$ correlates with $\hat e$ ⇒ the response reduces to
  $R_{\rm self}$, i.e. `Gold-v1.md`'s $\langle R_{\rm flow}\rangle=0.2930$.

### 2.5 Literature

Fisher (1925) and Louis (1982) — score and observed information for latent-variable models (the EM
machinery). Bartlett identities / information equality. Glynn (1990), Williams (1992) — score-function
gradient estimator. **BFD** (Bernstein & Armstrong 2014; Bernstein et al. 2016) — exactly this in weak
lensing: $\gamma$ from $\sum\partial\log P$ normalized by $\sum\partial^2\log P$. **lensfit** (Miller
et al. 2013) — the "shear sensitivity" $\langle\partial e/\partial\gamma\rangle$ as a posterior
expectation, the first practical version. **metacalibration / metadetection** (Sheldon & Huff 2017;
Sheldon et al. 2020) — the *other* route to a data-driven response: finite-difference the images
rather than the model; prior-free, and metadetection's re-detection step is the image-space analogue
of shearing the whole scene. One level up in cosmology: score compression (Alsing & Wandelt 2018),
IMNN (Charnock, Lavaux & Wandelt 2018).

---

## 3. Where $R_{\rm blend}$ enters

Because $S_\gamma$ acts on the **whole scene**, the analytic generator splits by channel:

$$s = s_{\rm shape} + s_{\rm size} + s_{\rm flux} + s_{\rm nbr},$$

so the response of the measured shape is

$$R=\underbrace{\mathrm{Cov}(\hat e,\,s_{\rm shape})}_{R_{\rm self}}
\;+\;\underbrace{\mathrm{Cov}(\hat e,\,s_{\rm nbr})}_{R_{\rm blend}}.$$

**They combine by covariance, not by hand-summation.** This is the answer to "where does $R_{\rm blend}$
enter a Bayesian inference": it is not a separate additive term you bolt on, it is the neighbour
channel of the same score, and the combination is automatic.

**But the formalism does not let us off the hook.** $\mathrm{Cov}(\hat e, s_{\rm nbr})\equiv 0$ unless
$p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ actually **depends on neighbour geometry** — separation,
pair position angle, neighbour shape — the quantities shear distorts in a spin-2 way. If the flow
conditions only on *scalar* neighbour fluxes, the neighbour part of the score is uncorrelated with
$\hat e$ and that term vanishes identically.

**Current status.** The certified flow conditions on `nbr_flux_near/far/max` only — scalars, no
geometry. Hence $\mathrm{Cov}(\hat e,s_{\rm nbr})=0$ for it, and that is precisely *why* `Gold-v1.md`
must supply an external BlendEMU $R_{\rm blend}=0.1593$ and add it:
$m=R_{\rm sim}/(R_{\rm flow}+R_{\rm blend})-1$. The additive form is a symptom of geometry-blindness,
not a design choice.

The geometry-aware option already exists in the repo: `sbs_shear/forward_model.py`
`SetConditionedForwardModel` with `sbs_shear/scene_model.py`'s `full` feature set
(`DEFAULT_SCENE_NEIGHBOR_FEATURES`: `distance_scaled`, `pair_pframe_cos2/sin2`,
`e_pframe_parallel_s/cross_s`) versus the angle-marginalized `radial` set. Its known cost is the
DeepSets trunk's isolated self-response deficit, which is why a **hybrid** (tabular mean head for the
primary self-response, set-conditioned residual for the blend channel) is the natural shape of a fix.

---

## 4. Selection

### 4.1 Why the posterior mean is structurally blind

The per-object posterior mean

$$\tilde e(\hat{\mathbf{x}})=\frac{\int e\;p(\hat{\mathbf{x}}\mid e,\cdot)\,p_0(e)\,de}
{\int p(\hat{\mathbf{x}}\mid e,\cdot)\,p_0(e)\,de}$$

is **normalized per object, over $e$**. Nothing about which objects entered the catalogue appears in
it. Add any cut $W$ and every surviving galaxy's $\tilde e$ is bit-identical to what it was without
the cut; only the ensemble you average over changes.

The trap is that the tower rule still holds. Since $W$ is $\hat{\mathbf{x}}$-measurable,

$$\mathbb E\big[\tilde e \,\big|\, \hat{\mathbf{x}}\in S\big]
=\mathbb E\big[e_{\rm true}\,\big|\,\hat{\mathbf{x}}\in S\big]\qquad\textbf{exactly.}$$

So $\tilde e$ **is** an unbiased estimate of the selected sample's true shapes — correct, and useless,
because the selected sample's true shapes are no longer an unbiased tracer of shear.

**Empirical proof (cont.154).** A cut built from a genuinely shear-invariant property gives selection
bias $+0.0000$ at every threshold; the same cut built from the *observed* (sheared) shape gives a
non-zero bias, because $\mathrm{Cov}(e_t,\text{pass})\neq0$ — aligned galaxies are elongated, lose
S/N, and drop out. That covariance does not cancel over orientations even though $\langle e\rangle$
does. Corollary: the true-`Re` null is exactly $0$, as it must be.

**Conclusion.** The posterior removes *measurement* bias and leaves *selection* bias untouched.
Selection lives entirely in the likelihood's normalization — exactly the thing $\tilde e$ divides out.

### 4.2 The selected-sample response

$$\langle\hat e\rangle_S(\gamma)=\frac{\mathbb E_\gamma[\hat e\,W]}{\mathbb E_\gamma[W]} .$$

Apply the quotient rule at $\gamma=0$. For an isotropic unsheared population and a rotationally
invariant cut (any cut on $\hat s$, $\hat m$, S/N — **not** a cut on $\hat e_1$ alone), the numerator
vanishes, $\mathbb E_0[\hat e W]=0$, killing the second term. What survives is §2.2 with
$f=\hat e W$:

$$\boxed{\;R_S=\frac{\mathrm{Cov}_0\big(\hat e\,W(\hat{\mathbf{x}}),\;s(\hat{\mathbf{x}})\big)}{\mathbb E_0[W]}\;}$$

The only thing selection changed was $f:\hat e\to\hat e W$. $\mathbb E_0[W]$ is the kept fraction,
converting a sum-over-all into a mean-over-selected.

### 4.3 Selection response = the size/flux channel, made visible

Split the score again:

$$R_S\,\mathbb E[W]=\mathbb E[\hat e W s_{\rm shape}]+\mathbb E[\hat e W s_{\rm size}]
+\mathbb E[\hat e W s_{\rm flux}]+\mathbb E[\hat e W s_{\rm nbr}].$$

With **no cut** ($W\equiv1$) the size and flux terms vanish — $\hat e$ is uncorrelated with an
isotropic dilation. With a cut, $W$ correlates $\hat e$ with size, and
$\mathbb E[\hat e W s_{\rm size}]\neq0$. **That term is the selection response.** It is not a new
physical effect and not a new model; it is the size channel of the same score, switched on by the cut.

### 4.4 The boundary form — where the physics lives

Follow objects rather than the density.
$\partial_\gamma W=\nabla_{\hat{\mathbf{x}}}W\cdot\partial_\gamma\hat{\mathbf{x}}$, and for a hard
threshold $\nabla_{\hat s}W=\delta(\hat s-s_c)$:

$$\boxed{\;R_{\rm sel}=\frac{p(\hat s=s_c)}{P_{\rm pass}}\;
\Big\langle\,\hat e\;\partial_\gamma\hat s\,\Big\rangle_{\hat s=s_c}\;}$$

**Selection response is a boundary integral**: density at the cut edge $\times$ the correlation
between measured shape and the shear-response of the measured cut variable, evaluated *on the edge*.
Interior galaxies contribute nothing.

Note what survives: any **isotropic** part of $\partial_\gamma\hat s$ contributes
$\langle\hat e\rangle_{\rm boundary}\times\langle\partial_\gamma\hat s\rangle=0$. **Only the spin-2,
orientation-correlated part matters.**

### 4.5 The size response is purely spin-2

For the second-moment matrix under lensing ($A$ maps image→source, $\kappa=0$, $A^{-1}\approx I+G$
with $G$ symmetric traceless):

$$M_{\rm img}=A^{-1}M_{\rm src}A^{-\mathsf T}
\quad\Longrightarrow\quad
T\to T\,(1+2\,e\!\cdot\!g)
\quad\Longrightarrow\quad
\boxed{\;\frac{\partial\log T}{\partial\gamma_i}=2\,e_i\;}$$

where $e$ here is the **distortion** ellipticity,
$M=\tfrac T2\begin{pmatrix}1+e_1&e_2\\e_2&1-e_1\end{pmatrix}$ (related to ngmix `g` by
$e=2g/(1+|g|^2)$).

*Sign check:* $g_1>0$ stretches along $x$; a galaxy with $e_1>0$ has its major axis along $x$, so it
grows. ✔

The shear response of log-size is **twice the ellipticity** — zero isotropic part, maximally
correlated with $\hat e$ by construction. Substituting into §4.4:

$$R_{\rm sel}\;\approx\;\frac{p_c}{P_{\rm pass}}\;\cdot\;2\,\big\langle \hat e\,e\big\rangle_{\rm boundary}
\;\sim\;\frac{p_c}{P_{\rm pass}}\cdot 2\langle e^2\rangle\,R_{\rm meas}.$$

Large and non-vanishing.

### 4.6 Sign predictions versus measurement (cont.156)

| cut | mechanism | predicted sign | measured shift |
|---|---|---|---|
| `size > 3.5` | keeps big ones; aligned galaxies grow ⇒ over-represented | **+** | **+9.30 %** |
| `mag < 24.5` | $\kappa=0$ ⇒ no first-order flux magnification (corroborated: cont.149 measures the mag coupling $b_{\rm mag}\approx0$, "flux conserved, as physics dictates"), but `mag_auto`'s size-scaled Kron aperture inherits the size response ⇒ also spin-2; aligned galaxies measure brighter ⇒ kept | **+** | **+13.79 %** |
| true-`Re` cut | shear-invariant property ⇒ $\partial_\gamma\hat s=0$ | **0 exactly** | **+0.0000** |

Both non-trivial signs and the null are predicted by the formula.

### 4.7 What the flow supplies, and why V2 was required

Because the cut is a function of the flow's **own output variables**, the selection function is
*derived*, not learned:

$$P_{\rm pass}(\mathbf{x},\mathbf{n})=\int_S p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,d\hat{\mathbf{x}}$$

— Monte Carlo over `bundle.sample`: draw $n_{\rm samples}$, count the fraction landing in $S$. No new
model, no new training. It enters the marginal likelihood's normalization, and in score form is just

$$s_{\rm sel}=s-\langle s\rangle_{\rm selected},$$

i.e. the Sheldon–Huff selection response is "center the score over the selected sample."

**Why V1 could not do this.** V1 takes measured mag/size as *inputs* and outputs shape only, so the
cut variables never move with shear and $P_{\rm pass}$ does not exist. V2/S2's 4D output
$(\hat e_1,\hat e_2,\hat m,\hat s)$ makes them endogenous. The same structural change that killed the
errors-in-variables floor (`memory/project_framing`) is what unlocks selection — not a coincidence,
the same fact.

**Detection is a different problem.** Undetected objects have no $\hat{\mathbf{x}}$; you cannot
integrate the flow over a region of output space that does not exist. That genuinely needs the
separate classifier (`sbs_shear/selection_model.py`). cont.153 confirms the certified constgold
catalogue is exactly the both-legs-detected intersection (29,811,214 of ~30.07 M per leg), so the
numbers above cleanly *exclude* detection — correct bookkeeping, not a solution.

---

## 5. How we compute it with our models

### 5.1 Level 1 — transport (already implemented)

The score form needs $\nabla\log p_0$, which we do not have. **We never need it for calibration**,
because of the equivalence

$$\mathrm{Cov}_0(\hat e W,\,s)\;=\;\partial_\gamma\,\mathbb E_\gamma[\hat e W],$$

and the right-hand side is evaluated by **moving samples, not differentiating a density**. The
catalogue's true properties *are* an empirical draw from $p_0$:

1. Draw $(\mathbf{x},\mathbf{n})$ from the catalogue.
2. Apply $S_{\pm\gamma}$ **analytically to truth**:
   `sbs_shear/shear_map.py::apply_shear_to_ellipticity` for primary and neighbour shapes,
   $\log T \mathrel{+}= 2e\!\cdot\!\gamma$, sheared separation vectors.
3. Push both legs through the flow's conditioners and sample with **common random numbers** (same
   latents both legs — this is what makes the difference low-noise).
4. Apply $W$ to the flow's *outputs*.
5. $R_S^{\rm model}=\big[\langle\hat e\rangle_S(+\gamma)-\langle\hat e\rangle_S(-\gamma)\big]/2\gamma$;
   compare with $R_S^{\rm sim}$ from the paired legs; $m=R_{\rm sim}/R_{\rm model}-1$.

**This is `scripts/eval_selection_response.py` (branch `ablation-v1-to-v2`, driver
`jobs/job_s2_selrecover_fluxsize.sh`), and cont.156 is this identity evaluated by transport.** The
math in §2–§4 is therefore not a new pipeline — it is the *theory of the pipeline already running*.
Its value is diagnostic.

### 5.2 Level 2 — factorize the failure

cont.156 result: the flow reproduces the truth selection shift on every mag cut and mild size cuts
(residual $m$ stays at the $\pm0.8$–$1.2\%$ no-cut floor), with **one failure**: `size>4.4`
over-shifts $+10.55\%$ against truth $+4.11\%$ ⇒ $m=-6.71\%$.

Apply §4.4. cont.156 also reports `fracS = fracM = 0.43` — the kept fraction matches exactly, so
$p_c/P_{\rm pass}$ is right. **Therefore the entire $-6.71\%$ sits in
$\langle\hat e\,\partial_\gamma\hat s\rangle$ at the boundary.** Measure it directly instead of
inferring it from an end-to-end residual:

- restrict to a thin shell $|\hat s-s_c|<\delta$;
- **model side:** $\langle\hat e\,(\hat s_+-\hat s_-)/2\gamma\rangle$ from the flow's $\pm$ legs with CRN;
- **truth side:** the same, from `det_meas` $g=0\leftrightarrow g=0.02$ both-detected matched pairs,
  which carry per-leg measured mag and `flux_radius`.

### 5.3 Why a mean-only pin cannot close it

Expand the failing correlation using the `ConditionalMeanFlow` structure:

$$\partial_\gamma\hat s=\underbrace{\frac{\partial\mu_{\hat s}}{\partial\log T}}_{\textstyle c(\mathbf{x})}\!\cdot 2e
\;+\;\frac{\partial\mu_{\hat s}}{\partial e}\cdot v_e\;+\;\text{residual reshaping},$$

$$\Longrightarrow\qquad
\big\langle\hat e\,\partial_\gamma\hat s\big\rangle\approx 2\big\langle \hat e\,e\,c(\mathbf{x})\big\rangle.$$

The theta-coupling pin (cont.149) constrains the population **mean** $\bar c$ — it moved the size
coupling from $-0.502$ (wrong sign) to $+0.563$ against truth $+0.566$. Selection needs $c$
**weighted by $\hat e\,e$ at the boundary**. If $c$ varies across the population then
$\langle \hat e\,e\,c(\mathbf{x})\rangle\neq\bar c\,\langle \hat e\,e\rangle$.

**Two matched first moments, one broken second moment.** A mean-only pin is structurally blind to the
quantity selection depends on.

*Directly testable:* plot $c(\mathbf{x})=\partial\mu_{\hat s}/\partial\log T$ against measured size —
model side free by autograd through the mean head, truth side from matched pairs binned by size. The
formula predicts the flow is monotonic where truth turns over past $\hat s\approx4$, which is exactly
the non-monotonic truth shift cont.156 observed (peaks at `>3.5`, drops at `>4.4`) and the flow
missed. This independently confirms cont.156's own proposed fix — Direction-A true-size perturbation
so the size response and the shape-correlation emerge self-consistently, replacing the mean-only pin.

### 5.4 Technical caveat — the location-family shortcut does not extend to size

`flow_drop_indices=[0,1,8,9]` drops **only** shape (`e1_input_p`, `e2_input_p` and their missing
flags). So:

- **shape axis:** the residual flow is blind to $e$ ⇒ location family ⇒ shearing the true shape moves
  only the mean head $\mu$ ⇒ exact grid, cheap, no resampling;
- **size axis:** the residual flow *does* see size ⇒ shearing true size reshapes the whole conditional
  density, not just its mean ⇒ **no $\mu$-only shortcut**; real sampling / `log_prob` is required.

Any estimator built on the size axis pays this cost.

### 5.5 Level 3 — what a full Bayesian estimator would additionally need

Only the per-object optimal-weight form $\hat\gamma=\sum_i s_i/\sum_i\mathcal I_i$ (and its Fisher
error bars) requires $s$ per object, hence $\nabla\log p_0$ — a real intrinsic-property prior over
shape, size, flux, Sérsic and neighbour configuration. Fitting it to the simulation catalogue is fine
for closure tests but circular for science; `SBI_shear.md:189` demands an external deep-field prior.
**This is the genuine cost of Direction B, and it is not needed for anything in §5.1–§5.3.**

---

## 6. Honest limits

- **`size>4.4`, $m=-6.71\%$** — open. Marginals correct (kept fraction matches), shape–size copula
  wrong in the tail. This is the hard part, and §5.2 says which single number to measure.
- **The flux/size shear response is supervised, not emergent.** It was *sign-flipped* until the
  theta-coupling pin. Selection calibration is downstream of a quantity NLL training does not learn
  on its own.
- **Geometry-blind blending kills $\mathrm{Cov}(\hat e,s_{\rm nbr})$** (§3). If neighbours contaminate
  measured flux/size, the *oriented* part of blend-induced selection is also invisible.
- **Selection is a boundary effect, so it amplifies prior sensitivity.** The response is set by
  galaxies at the cut edge; global NLL says nothing about the edge, and deep-field prior mismatch
  there propagates straight through.
- **"Response from data" means "no calibration sims", not "model-free".** Bartlett returns the
  response *of your model*; misspecification shows up as $m\neq0$ in exactly that amount. Paired-sim
  $R_{\rm sim}$ remains the test, not a redundancy.
- **Only cuts on modelled outputs.** A cut on colour, a morphology flag, or a blendedness metric the
  flow does not output has no handle at all.

---

## 7. Consistency checks this document must pass

| formula | prediction | measured | source |
|---|---|---|---|
| §4.4 + §4.5, `size>3.5` | shift $>0$ | $+9.30\%$ | cont.156 |
| §4.4 + §4.5, `mag<24.5` | shift $>0$ | $+13.79\%$ | cont.156 |
| §4.4, shear-invariant cut | exactly $0$ | $+0.0000$ | cont.154 control, true-`Re` null |
| §2.4, noiseless limit | $\mathcal R=2(1-e_{\rm rms}^2)$; $\to1$ for $\varepsilon$ | — | BJ02; `SBI_shear.md:262-267` |
| §2.4, no cut ⇒ all $s_{\rm shape}$ | $R=R_{\rm self}$ | $\langle R_{\rm flow}\rangle=0.2930$ | `Gold-v1.md:96` |
| §3, geometry-blind ⇒ $\mathrm{Cov}(\hat e,s_{\rm nbr})=0$ | $R_{\rm blend}$ must be external and additive | $R_{\rm blend}=0.1593$ added by hand | `Gold-v1.md:95` |

---

## 8. Pointers

- Framework spec: `SBI_shear.md` (§2 prior shift, §3 conditioning, prior-dependence discussion).
- Certified numbers: `Gold-v1.md` ($R_{\rm sim}=0.4534$, $\langle R_{\rm flow}\rangle=0.2930$,
  $R_{\rm blend}=0.1593$, $m=+0.245\%$); `Gold-V2.md` (Direction A vs B).
- Measured selection results: `WORKLOG.md` cont.153 (detection pairing), cont.154 (shear-invariant
  control), cont.155 (true vs measured S/N), cont.156 (flux/size recovery + the `size>4.4` gap).
- Code: `sbs_shear/posterior_shape.py` (posterior grid), `sbs_shear/shear_map.py` (analytic
  $S_\gamma$), `sbs_shear/measurement_model.py` (`ConditionalMeanFlow`, `flow_drop_indices`),
  `sbs_shear/forward_model.py` + `sbs_shear/scene_model.py` (geometry-conditioned option),
  `sbs_shear/selection_model.py` (detection classifier),
  `scripts/eval_selection_response.py` / `scripts/eval_selection_constgold.py` (branch
  `ablation-v1-to-v2`).

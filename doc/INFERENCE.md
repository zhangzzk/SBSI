# INFERENCE.md — shear, response, and selection from the forward model

Derivations and algorithms only. Measured results, model status and project history live in
`WORKLOG.md`, `Gold-V1.md`, `Gold-V2.md` and `MILESTONE.md`; the frozen V3 numbers are in
`MILESTONE.md`. The older framework spec is `archive/pre-v3/docs/SBI_shear.md` (provenance —
it predates the V3 library and is no longer maintained).

**Summary.** The shear response is a **covariance with the score**. Selection bias is a **boundary
term built from that same score**, dominated by its size channel. The blend response is its
**neighbour channel**. All three follow from one identity, and none requires a separate model.

New to the score/information machinery? **Appendix A** derives it from scratch on a one-dimensional
Gaussian, with a dictionary between the statistics names and the lensing ones. §§1–7 do not depend
on it.

---

## 0. Implementation status — what the shipped library actually does

This section is the shipped-code boundary that `MILESTONE.md` and
`sbsi/posterior_shape.py` point at. Everything from §1 onward is the derivation; it
describes the target, not the current implementation.

SBSI can load an explicit measurement-flow checkpoint and evaluate its
four-dimensional detected-object likelihood over a latent ellipticity grid.
`BayesianInference.load(checkpoint)` and its catalogue arguments are
model-name agnostic and accept user-owned paths or DataFrames.

The response input-catalogue boundary is implemented independently in
`sbsi.forward_catalogue`: SBSI validates a truth catalogue and returns an
aligned one-row-per-primary flow view plus a many-row primary/secondary emulator
view. It computes the flow's intrinsic-shape and crowding features, applies the
emulator's recorded pair cuts and rescaling, and preserves `primary_row` keys.
BlendEMU is called only to evaluate its trained model on the prepared pair
table. This preparation completes the two inputs for response prediction, but
it does not create the measured-target catalogue or complete the joint shear
likelihood described below.

This is not yet the validated catalogue-level shear inference requested for the
public workflow. That implementation still needs a joint generative likelihood
for the measurement flow and neighbour/emulator contribution, detection and
selection normalization, latent scene marginalization, a hierarchical population
prior, ensemble uncertainty, and simulation-based coverage tests.

Until those pieces are complete, the tutorial leaves Part 3 as TODO. Existing
posterior-grid and empirical-Bayes code is a research prototype for a detected
population and must not be presented as the final simulation-based shear result.

---

## 1. Setup and notation

| symbol | meaning |
|---|---|
| $\mathbf{x}$ | true properties of the primary: shape $e$, size $T$, flux, Sérsic index |
| $\mathbf{n}$ | true neighbour configuration: separations, position angles, shapes, sizes, fluxes |
| $\hat{\mathbf{x}}$ | measured catalogue quantities; for a 4D flow $\hat{\mathbf{x}}=(\hat e_1,\hat e_2,\hat m,\hat T)$ |
| $T$, $\hat T$ | true and measured **size**. The bare letter $s$ is never a size — it is the score |
| $e$, $\varepsilon$ | **distortion** and **reduced-shear** ellipticity of the true shape, $e=2\varepsilon/(1+\lvert\varepsilon\rvert^2)$; ngmix `g` is $\varepsilon$. They differ by $\approx2$, so every velocity in §2.4 names its convention |
| $p_\gamma(\mathbf{x},\mathbf{n})$ | the **sheared prior on truth**, $p_\gamma\equiv p_0\circ S_{-\gamma}$ — not to be confused with $p(\hat{\mathbf{x}}\mid\gamma)$, the marginal on the **data** side |
| $s(\hat{\mathbf{x}})$ | the **score**, (2.1); $s_i\equiv s(\hat{\mathbf{x}}_i)$, and $s_{\rm shape},s_{\rm nbr},\dots$ its channels (3.1) |
| $u(\mathbf{x},\mathbf{n})$ | the analytic **generator** on the truth side, (2.7); $s=\mathbb E_{\rm post}[u]$ by (2.2) |
| $p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ | the learned flow — **shear-free** |
| $p_0$ | intrinsic (unsheared) prior over the whole scene |
| $S_\gamma$ | analytic shear map, acting on **truth** |
| $W(\hat{\mathbf{x}})$ | cut indicator, e.g. $\mathbb 1[\hat T>T_c]$ |
| $i=1\ldots N$ | index over **objects** in the catalogue: $\hat{\mathbf{x}}_i$ is one measured galaxy |
| $k$ | index over **prior nodes** $(\mathbf{x}_k,\mathbf{n}_k)\sim p_0$ (§5B), shared by every object |

Shear appears in exactly one place — it shifts the prior:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\;
p_0\big(S_{-\gamma}(\mathbf{x},\mathbf{n})\big)\,d\mathbf{x}\,d\mathbf{n}. \tag{1.1}$$

Write $p_\gamma(\mathbf{x},\mathbf{n})\equiv p_0\big(S_{-\gamma}(\mathbf{x},\mathbf{n})\big)$ for that
sheared prior. Two densities in this document carry a $\gamma$ and they live on opposite sides of the
model: $p_\gamma$ is on **truth** and is analytic, while $p(\hat{\mathbf{x}}\mid\gamma)$ is on
**data** and is the intractable marginal. Everything below follows from this structure alone.

---

## 2. The score, and three identities

$$s_\gamma(\hat{\mathbf{x}})\;\equiv\;\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma),
\qquad s\;\equiv\;s_0. \tag{2.1}$$

$s$ — no subscript — is the score **at zero shear**, a fixed function of the data used everywhere
below. The family $s_\gamma$ is needed only when something must be differentiated in $\gamma$ a
second time, as in (2.4).

### 2.1 Fisher's identity

Differentiating (1.1),

$$\partial_\gamma p(\hat{\mathbf{x}}\mid\gamma)=\int p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,
p_\gamma\,\partial_\gamma\log p_\gamma\;d\mathbf{x}\,d\mathbf{n},$$

and dividing by $p(\hat{\mathbf{x}}\mid\gamma)$ turns the bracket into the posterior:

$$\boxed{\;\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)
=\mathbb E_{p(\mathbf{x},\mathbf{n}\mid\hat{\mathbf{x}})}\big[\partial_\gamma\log p_\gamma(\mathbf{x},\mathbf{n})\big]\;}\tag{2.2}$$

The bracketed factor is analytic, because shear enters truth through the closed-form $S_\gamma$.
**The flow is never differentiated with respect to $\gamma$** — $\gamma$ is not one of its inputs.

### 2.2 The covariance identity

For any fixed statistic $f(\hat{\mathbf{x}})$ — the function is fixed, only the density moves:

$$\partial_\gamma\mathbb E_\gamma[f]=\int f\,\partial_\gamma p=\int f\,p\,\partial_\gamma\log p
=\mathbb E_\gamma[f\,s_\gamma],$$

and evaluating at $\gamma=0$, where $s_0=s$ and $\mathbb E_0[s]=0$,

$$\boxed{\;\partial_\gamma\mathbb E_\gamma[f]\big|_0=\mathrm{Cov}_0\big(f(\hat{\mathbf{x}}),\,s(\hat{\mathbf{x}})\big)\;}\tag{2.3}$$

(the likelihood-ratio / score-function estimator). **(2.3) is the spine of this document.** Sections
3 and 4 are not new derivations; they are choices of $f$:

| choose $f=$ | and (2.3) returns | § |
|---|---|---|
| $\hat e$ | $R=\mathrm{Cov}_0(\hat e,s)$, the shear response | 3 |
| $\hat e$, with $s$ split by channel | $R_{\rm self}+R_{\rm blend}$, combined by covariance rather than by hand | 3.1–3.2 |
| $\hat e\,W$ | $R_S=\mathrm{Cov}_0(\hat eW,s)/\mathbb E_0[W]$, the selected-sample response | 4.2 |
| $s$ itself | $\mathcal I=\mathrm{Var}_0[s]$, the self-calibration | 2.3 |

The last row is the outlier, and that is the whole point: for every other $f$ the right-hand side is
a number that must be supplied from elsewhere, whereas for $f=s$ it is the scatter of the numbers
already in hand.

**Evaluating it.** Since $\mathbb E_0[s]=0$ the covariance is a bare product average, so every row
above is one pass over the catalogue:

$$\mathrm{Cov}_0(f,s)\;\approx\;\frac1N\sum_i f_i\,s_i,
\qquad f_i\equiv f(\hat{\mathbf{x}}_i),\quad s_i\equiv s(\hat{\mathbf{x}}_i).$$

So the shear response is $R\approx N^{-1}\sum_i\hat e_i\,s_i$ — no paired $\pm\gamma$ renders and no
finite differences. All of the cost sits in producing the per-object $s_i$ (§5B); everything after
that is arithmetic on those numbers.

### 2.3 Bartlett and Louis

$$\mathbb E_0[s]=0,\qquad
\mathcal I\equiv\mathrm{Var}_0[s]=-\mathbb E_0\big[\partial_\gamma s_\gamma\big|_0\big],\tag{2.4}$$

The subscripts are essential, and there are two kinds. $s=s_0$ is a **fixed function of the data**;
$\mathbb E_0$ versus $\mathbb E_\gamma$ says only which distribution that fixed function is averaged
over. The $\partial_\gamma s_\gamma$ in the second identity is the *only* place the $\gamma$-family of
(2.1) is needed — differentiate first, then set $\gamma=0$. So the two statements

$$\mathbb E_0[s]=0
\qquad\text{and}\qquad
\mathbb E_\gamma[s]=\mathcal I\gamma+O(\gamma^2)$$

do not conflict — the second reduces to the first at $\gamma=0$ — and together they *are* the
measurement: the score averages to zero in an unsheared universe, so its departure from zero in a
sheared one estimates $\gamma$, with $\mathcal I$ as the exchange rate.

The second identity in (2.4) is just (2.3) applied to $f=s$: **the score's own shear response is its
own variance.** Every other statistic needs its response supplied from outside; this one carries its
calibration in the scatter of the same numbers being summed. Concretely, the product average of §2.2
at $f=s$ is a sum of squares,

$$\mathcal I\;\approx\;\frac1N\sum_i s_i^{\,2},$$

so the estimator below needs no machinery beyond two accumulators over the catalogue:
$\hat\gamma=\sum_i s_i\big/\sum_i s_i^{\,2}$.

Both expectations in (2.4) are at $\gamma=0$, so $\mathcal I$ is a **population** quantity. Its
per-object counterpart is the *observed information*

$$\mathcal I_i\;\equiv\;-\partial^2_\gamma\log p(\hat{\mathbf{x}}_i\mid\gamma)\big|_0,
\qquad s_i\equiv s(\hat{\mathbf{x}}_i),$$

the curvature of the log-likelihood of the single measurement $\hat{\mathbf{x}}_i$. Since
$\partial_\gamma s_\gamma=\partial^2_\gamma\log p$, the second identity in (2.4) is precisely the
statement

$$\mathcal I=\mathbb E_0\big[\mathcal I_i\big]$$

— the population Fisher information *is* the average observed information — so $\sum_i\mathcal I_i$
estimates $N\mathcal I$ without bias. Louis (1982) makes $\mathcal I_i$ computable from the posterior
of §2.1 alone, with no second set of samples:

$$\partial^2_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)
=\mathbb E_{\rm post}\big[\partial^2_\gamma\log p_\gamma\big]
+\mathrm{Var}_{\rm post}\big[\partial_\gamma\log p_\gamma\big].\tag{2.5}$$

(2.5) reads *observed information = complete-data information $-$ missing information*: the
subtracted $\mathrm{Var}_{\rm post}[u]$ is the disagreement, among the truths still compatible with
$\hat{\mathbf{x}}_i$, about which way shear points. Hence, summing over the $N$ objects,

$$\hat\gamma=\frac{\sum_i s_i}{\sum_i\mathcal I_i}
\qquad\Longrightarrow\qquad
\partial_\gamma\mathbb E[\hat\gamma]=1\ \text{ by construction},\tag{2.6}$$

which is nothing but the inversion of the straight line $\mathbb E_\gamma[s]=\mathcal I\gamma$: the
numerator grows as $N\mathcal I\gamma$, the denominator estimates $N\mathcal I$, so the same
$\mathcal I$ appears above and below and cancels. There is no external responsivity applied
afterwards and no $R_{\rm self}+R_{\rm blend}$ summed by hand. By Cramér–Rao (2.6) is the
minimum-variance first-order estimator, and the per-object weight is its own Fisher information
rather than a tuned quantity.

Both $\sum_i s_i^{\,2}$ and $\sum_i\mathcal I_i$ estimate $N\mathcal I$, so either may serve as the
denominator of (2.6). They differ in noise: $s_i^{\,2}$ uses only the single realized vote, whereas
Louis's $\mathcal I_i$ uses the whole posterior spread behind that vote and is generally the quieter
estimate. Their agreement is a free internal consistency test.

Everything in this subsection is stated for a **complete** sample. Under a cut the numerator and the
denominator of (2.6) each acquire the *same* correction, and dropping the second one biases
$\hat\gamma$; §5B.2 derives both.

### 2.4 The generator

Let $v\equiv\partial_\gamma S_\gamma|_0$ be the shear velocity field on truth. Since $p_\gamma$ is the
pushforward of $p_0$ along $S_\gamma$, the continuity equation
$\partial_\gamma p_\gamma+\nabla\!\cdot\!(p_\gamma v)=0$ gives

$$\boxed{\;u(\mathbf{x},\mathbf{n})\;\equiv\;\partial_\gamma\log p_\gamma\big|_0
\;=\;-\Big(v\cdot\nabla\log p_0+\nabla\!\cdot\! v\Big)\;}\tag{2.7}$$

The channel velocities, all closed-form:

| channel | velocity $v$ | order |
|---|---|---|
| shape (**reduced-shear** $\varepsilon$, ngmix `g`) | $v_\varepsilon=\partial_g\!\left[\dfrac{\varepsilon+g}{1+\bar g \varepsilon}\right]_0=\big(1-\varepsilon^2,\;i(1+\varepsilon^2)\big)$ for $(g_1,g_2)$ | spin-2 |
| log size (**distortion** $e$) | $v_{\log T}=2e$ — see §4.5 | spin-2, **no isotropic part** |
| log flux | $0$ at first order when $\kappa=0$; magnification $\mu=\big[(1-\kappa)^2-\lvert g\rvert^2\big]^{-1}$ otherwise | — |
| neighbour separation | $\delta r_i=\gamma_{ij}r_j$ | spin-2 in the pair position angle |
| neighbour shape | Möbius, as for the primary | spin-2 |

### 2.5 Limits

- **Noiseless measurement.** The posterior collapses to a delta, $s\to u$, and
  $\mathrm{Cov}(e,s)=\partial_\gamma\langle e\rangle$: the Bernstein & Jarvis
  responsivity $\mathcal R=2(1-e_{\rm rms}^2)$ in the distortion convention, $\mathcal R\to1$ for
  reduced shear (the isotropic average of $v_\varepsilon$ above is exactly $1$).
- **Low signal-to-noise.** The posterior tends to the prior, so $s\to\mathbb E_{p_0}[u]=0$: the object
  contributes nothing rather than contributing noise.
- **No cut.** The neighbour channel drops for a geometry-blind likelihood (§3), leaving
  $R_{\rm self}$ — the primary's own shape, size and flux channels.

### 2.6 Literature

Fisher (1925), Louis (1982) — score and observed information for latent-variable models.
Bartlett identities / information equality. Glynn (1990), Williams (1992) — score-function gradient
estimator. **BFD** (Bernstein & Armstrong 2014; Bernstein et al. 2016) — this construction in weak
lensing: $\gamma$ from $\sum\partial\log P$ normalized by $\sum\partial^2\log P$.
**lensfit** (Miller et al. 2013) — shear sensitivity $\langle\partial e/\partial\gamma\rangle$ as a
posterior expectation. **Bernstein & Jarvis (2002)** — the closed-form responsivity.
**metacalibration / metadetection** (Sheldon & Huff 2017; Sheldon et al. 2020) — the alternative
data-driven route: finite-difference the *images*, prior-free; metadetection's re-detection step is
the image-space analogue of shearing the whole scene. One level up: score compression
(Alsing & Wandelt 2018), IMNN (Charnock, Lavaux & Wandelt 2018).

---

## 3. Channel decomposition — $R_{\rm self}$ and $R_{\rm blend}$

$S_\gamma$ acts on the whole scene, so (2.7) splits:

$$s=s_{\rm shape}+s_{\rm size}+s_{\rm flux}+s_{\rm nbr},\tag{3.1}$$

and (2.3) is linear in $s$, so the response splits with it. Group by **which object** a channel
belongs to, not by which property:

$$R=\underbrace{\mathrm{Cov}\big(\hat e,\;s_{\rm shape}+s_{\rm size}+s_{\rm flux}\big)}_{R_{\rm self}}
+\underbrace{\mathrm{Cov}\big(\hat e,\;s_{\rm nbr}\big)}_{R_{\rm blend}}.\tag{3.2}$$

**The two combine by covariance, not by hand-summation.** There is no separate additive blend term to
bolt on; it is the neighbour channel of the same score.

**The size channel does not drop out.** It is tempting to keep only $s_{\rm shape}$ on the grounds
that $\hat e$ cannot correlate with a dilation, but $v_{\log T}=2e$ is spin-2 with *no* isotropic part
(§2.4, §4.5), so that argument does not apply. With $\partial_{\log T}v_{\log T}=0$ the size generator
is $u_{{\rm size},i}=-2e_i\,\psi$, where $\psi\equiv\partial_{\log T}\log p_0$. Let
$h(T)\equiv\mathbb E[\hat e_ie_i\mid T]$ be the shape-measurement fidelity at fixed size. Then, using
the tower rule and integrating by parts,

$$\mathrm{Cov}(\hat e_i,s_{{\rm size},i})=-2\,\mathbb E\big[\hat e_ie_i\,\psi\big]
=-2\!\int\! h\;\partial_{\log T}p_0\;d\log T
=2\,\mathbb E\!\left[\frac{\partial h}{\partial\log T}\right].\tag{3.3}$$

It vanishes **if and only if shape-measurement fidelity is independent of size**, which no real survey
satisfies. The physics is ordinary: shear grows aligned galaxies, larger galaxies suffer less noise
dilution, so they carry more measured shape. This is part of the primary's own response, which is why
(3.2) groups it into $R_{\rm self}$.

The flux channel *does* vanish, but for a stated reason rather than by symmetry: $v_{\log F}=0$ at
first order when $\kappa=0$ (§2.4). It returns with convergence.

**Non-degeneracy condition.** $\mathrm{Cov}(\hat e,s_{\rm nbr})\equiv 0$ unless
$p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ **depends on neighbour geometry** — separation, pair
position angle, neighbour shape — the quantities shear distorts with spin 2. A likelihood conditioned
only on *scalar* neighbour fluxes is invariant under rotating a neighbour about the primary at fixed
flux, so the neighbour score is uncorrelated with $\hat e$ and the term vanishes identically. See
§5B.1 for the same statement at the level of posterior weights.

**Consequence.** Under a geometry-blind likelihood, $R_{\rm blend}$ cannot be recovered from the
model and must be supplied externally and added, i.e. $m=R_{\rm sim}/(R_{\rm self}+R_{\rm blend})-1$
for the multiplicative bias $m$. The additive form is a symptom of the missing conditioning, not a
modelling choice.

**Model status — the condition is met, the implementation does not use it.** The two are separate
and it is worth stating which is which.

*Conditioning.* The Gold-v1 flow sees scalar neighbour fluxes only, so for it the term above vanishes
identically and the external $R_{\rm blend}=0.1593$ is forced. The V2 model is not that model:
`NEIGHBOR_FEATURES` (`scripts/train_joint_forward.py:76`) carries `distance_scaled`,
`relative_position_angle_cos2`/`sin2` and the neighbour shape `e1_input_s`/`e2_input_s`. Separation,
spin-2 pair orientation and neighbour shape are all present, so the non-degeneracy condition is
satisfied and $\mathrm{Cov}(\hat e,s_{\rm nbr})$ has somewhere to come from.

*Shear map.* The implemented map does not exercise it. `shifted_feature_frame`
(`scripts/train_joint_forward.py:227-247`) applies the Möbius transform to the primary's ellipticity
and — under `--shear-both` — the neighbour's, and to nothing else. `distance_scaled` is rebuilt from
the untouched `distance` and `Re_input_p`; `relative_position_angle_*` from the untouched
`polarization_angle` (`sbs_shear/preprocessing.py:166-179`, reached via `rescale`). True size is not
sheared either, so the $v_{\log T}=2e$ row of §2.4 is absent from this path. `scene_context`
(`scripts/closure_v2_lagrangian.py:112`) hardcodes `primary_only=True`, so the §5C closure runs with
the primary shape as the *only* moving input.

So the positional velocity of §2.4 is zero **by omission in code**, not by the degeneracy above. The
neighbour-shape channel is reachable today through the existing `--shear-both`; the positional
channel needs the separation vector sheared, which is a change to `shifted_feature_frame` rather than
to the architecture. §5B.1 shows the two channels are not interchangeable — they have different
sources and one of them nearly vanishes.

---

## 4. Selection

### 4.1 The posterior mean is structurally blind to selection

$$\tilde e(\hat{\mathbf{x}})=\frac{\int e\;p(\hat{\mathbf{x}}\mid e,\cdot)\,p_0(e)\,de}
{\int p(\hat{\mathbf{x}}\mid e,\cdot)\,p_0(e)\,de}\tag{4.1}$$

is normalized **per object, over $e$**. Nothing about catalogue membership enters it, so under any cut
every surviving galaxy's $\tilde e$ is unchanged; only the ensemble averaged over changes.

The tower rule then traps you. Since $W$ is $\hat{\mathbf{x}}$-measurable,

$$\mathbb E\big[\tilde e\,\big|\,\hat{\mathbf{x}}\in S\big]
=\mathbb E\big[e_{\rm true}\,\big|\,\hat{\mathbf{x}}\in S\big]\qquad\textbf{exactly},\tag{4.2}$$

so $\tilde e$ is an unbiased estimate of the *selected sample's* true shapes — correct, and useless,
because those shapes are no longer an unbiased tracer of shear. Selection lives entirely in the
likelihood's normalization, which (4.1) divides out.

The underlying mechanism: a cut computed from any **observed** (hence sheared) quantity gives
$\mathrm{Cov}(e_t,\text{pass})\neq0$, and that covariance does not cancel over orientations even
though $\langle e\rangle$ does. A cut on a genuinely **shear-invariant** quantity gives exactly zero.

### 4.2 The selected-sample response

$$\langle\hat e\rangle_S(\gamma)=\frac{\mathbb E_\gamma[\hat e\,W]}{\mathbb E_\gamma[W]}.$$

Apply the quotient rule at $\gamma=0$. For an isotropic unsheared population and a rotationally
invariant cut (on $\hat T$, $\hat m$, S/N — **not** on $\hat e_1$ alone) the numerator vanishes,
$\mathbb E_0[\hat eW]=0$, killing the second term. What remains is (2.3) with $f=\hat eW$:

$$\boxed{\;R_S=\frac{\mathrm{Cov}_0\big(\hat e\,W(\hat{\mathbf{x}}),\;s(\hat{\mathbf{x}})\big)}{\mathbb E_0[W]}\;}\tag{4.3}$$

Selection changed only $f:\hat e\to\hat eW$; the identity did not care. $\mathbb E_0[W]$ is the kept
fraction, converting a sum over all into a mean over the selected.

### 4.3 Every channel splits into an interior and a boundary piece

Substituting (3.1) into (4.3),

$$R_S\,\mathbb E[W]=\mathbb E[\hat eWs_{\rm shape}]+\mathbb E[\hat eWs_{\rm size}]
+\mathbb E[\hat eWs_{\rm flux}]+\mathbb E[\hat eWs_{\rm nbr}].\tag{4.4}$$

Each of those four terms has an exact Lagrangian counterpart, and the correspondence is worth making
explicit because §4.4 will split the *same* total a different way. Let $\gamma_c$ be a fictitious
parameter that switches on channel $c$'s velocity alone. Then $p_{\gamma_c}$ is a legitimate
one-parameter family whose generator is $u_c$ and whose score is $s_c$, so (2.3) applies to it by
itself; writing the same derivative pathwise — move the truth along $v_c$ at **fixed measurement
noise**, and the measurement moves with it — gives

$$\underbrace{\mathbb E\big[\hat eW\,s_c\big]}_{\text{Eulerian}}=\partial_{\gamma_c}\mathbb E[\hat eW]
=\underbrace{\mathbb E\big[W\,\partial_{\gamma_c}\hat e\big]}_{\textbf{interior}}
+\underbrace{\mathbb E\big[\hat e\,\nabla_{\hat{\mathbf x}}W\cdot\partial_{\gamma_c}\hat{\mathbf x}\big]}_{\textbf{boundary}}.
\tag{4.4b}$$

Both sides are the derivative of one number along one family, so (4.4b) is an **identity, not a
first-order approximation**: $f$ fixed with the density moving, versus objects moving with $f$
carried along.

So a cut does not *convert* a channel into a boundary term — it **adds a boundary piece to every
channel** and leaves the interior pieces in place, now averaged over survivors. With $W\equiv1$ the
boundary terms are absent, the flux interior term is zero at $\kappa=0$, and the size interior term is
the small population-averaged fidelity gradient (3.3). Summing (4.4b) over channels collects
$\sum_c\partial_{\gamma_c}\hat{\mathbf x}=\partial_\gamma\hat{\mathbf x}$, so the four boundary pieces
merge into the single term (4.5), written with the *total* $\partial_\gamma\hat T$.

**Why the size channel owns that boundary.** $R_{\rm sel}$ is the boundary sum over all channels, but
(4.6) gives the size channel's contribution to $\partial_\gamma\hat T$ in closed form, $2e$ — spin-2
at $O(1)$ and maximally aligned with $\hat e$ — whereas the shape and neighbour channels reach
$\hat T$ only through the measurement's own weaker size–shape and blending couplings, enumerated in
(5.1). §4.5 evaluates the size part on that expectation; §5A.2 is the diagnostic that tests it rather
than assuming it. **That is the precise sense in which selection bias is the size channel** — not new
physics and not a new model, but the boundary piece of the same score.

### 4.4 Boundary form

Following objects rather than the density,
$\partial_\gamma W=\nabla_{\hat{\mathbf{x}}}W\cdot\partial_\gamma\hat{\mathbf{x}}$, and for a hard
threshold $\nabla_{\hat T}W=\delta(\hat T-T_c)$:

$$\boxed{\;R_{\rm sel}=\frac{p_c}{P_{\rm pass}}\;
\big\langle\,\hat e\;\partial_\gamma\hat T\,\big\rangle_{\hat T=T_c}\;},
\qquad p_c\equiv p(\hat T=T_c)\tag{4.5}$$

$R_{\rm sel}$ and $R_S$ are **not** the same object: $R_S$ (4.3) is the total response of the selected
sample, while $R_{\rm sel}$ is only the boundary piece the cut creates. Collecting the *interior*
terms of (4.4b) by object,

$$R_S=R_{\rm self}\big|_S+R_{\rm blend}\big|_S+R_{\rm sel},
\qquad
R_X\big|_S\equiv\frac{\mathbb E\big[W\,\partial_\gamma^{(X)}\hat e\big]}{\mathbb E[W]},$$

where $\big|_S$ is the **interior (Lagrangian) restriction** — the survivors' own shapes moving — and
*not* the Eulerian covariance (3.2) evaluated on survivors. Keeping that distinction is what makes the
split exact rather than heuristic: (4.4b) removes the boundary piece from each channel *before* the
channels are regrouped, so every term of (4.4) is counted exactly once and $R_{\rm sel}$ does not
double-count with $R_{\rm self}|_S$. Conflating the two restrictions is the easy mistake here, and it
double-counts the size channel.

The selection response is a **boundary integral**: density at the cut edge times the correlation
between measured shape and the shear response of the cut variable, evaluated on the edge. Interior
objects contribute nothing.

Any **isotropic** part of $\partial_\gamma\hat T$ contributes
$\langle\hat e\rangle_{\rm boundary}\times\langle\partial_\gamma\hat T\rangle=0$. **Only the spin-2,
orientation-correlated part survives.**

### 4.5 The size response is purely spin-2

For the second-moment matrix under lensing, with $A$ mapping image$\to$source, $\kappa=0$, and
$A^{-1}\approx I+G$ ($G$ symmetric traceless):

$$M_{\rm img}=A^{-1}M_{\rm src}A^{-\mathsf T}
\quad\Longrightarrow\quad T\to T\,(1+2\,e\!\cdot\!g)
\quad\Longrightarrow\quad
\boxed{\;\partial\log T/\partial\gamma_i=2e_i\;}\tag{4.6}$$

with $e$ the **distortion** ellipticity,
$M=\tfrac T2\left(\begin{smallmatrix}1+e_1&e_2\\e_2&1-e_1\end{smallmatrix}\right)$, related to ngmix
`g` by $e=2g/(1+|g|^2)$. Sign check: $g_1>0$ stretches along $x$, and a galaxy with $e_1>0$ has its
major axis along $x$, so it grows.

The response of log-size is thus **twice the ellipticity** — zero isotropic part, maximally correlated
with $\hat e$. Substituting into (4.5),

$$R_{\rm sel}\approx\frac{p_c}{P_{\rm pass}}\cdot2\big\langle\hat e\,e\big\rangle_{\rm boundary}
\sim\frac{p_c}{P_{\rm pass}}\cdot2\langle e^2\rangle\,R_{\rm meas},\tag{4.7}$$

with $e$ the **distortion** of (4.6) — not the reduced-shear $\varepsilon$ of §2.4, a factor of
$\approx2$ — and $R_{\rm meas}\equiv\partial\langle\hat e\rangle/\partial e$ the measurement's own
shape response.

### 4.6 Consequences for cut direction

| cut | mechanism | predicted shift |
|---|---|---|
| keep large ($\hat T>T_c$) | aligned galaxies grow ⇒ over-represented | $>0$ |
| keep bright ($\hat m<m_c$) | at $\kappa=0$ no first-order flux magnification, but a size-scaled aperture (e.g. Kron) makes $\hat m$ inherit the size response ⇒ also spin-2; aligned galaxies measure brighter ⇒ kept | $>0$ |
| cut on a shear-invariant true property | $\partial_\gamma\hat T=0$ | $0$ exactly |

### 4.7 The selection function, and the 4D-output requirement

The cut is a function of the flow's **own output variables**, so the selection function is *derived*,
not learned:

$$P_{\rm pass}(\mathbf{x},\mathbf{n})=\int_S p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,d\hat{\mathbf{x}}
\tag{4.8}$$

— Monte Carlo: draw $n_{\rm samples}$ from the flow, count the fraction inside $S$. It enters the
marginal likelihood's normalization, and in score form is a single centering,

$$s_{\rm sel}=s-\langle s\rangle_{\rm sel}\tag{4.9}$$

— the score-space analogue of the Sheldon–Huff selection response, which is stated in $\hat e$. The
centering is only half of it: truncation rescales the log-likelihood's *curvature* as well as shifting
its slope, so $\mathcal I$ carries a matching correction (§5B.2). Applying (4.9) alone and leaving the
denominator of (2.6) untouched leaves a multiplicative bias equal to the fractional information the
cut removes.

**Requirement.** (4.8) exists only if the flow **outputs** the cut variables. A model that takes
measured mag/size as *inputs* and outputs shape alone has no $P_{\rm pass}$: its cut variables never
move with shear. A 4D output $(\hat e_1,\hat e_2,\hat m,\hat T)$ makes them endogenous.

**Status (2026-08-17).** Met in code. The V3 checkpoint outputs
$(\hat e_1,\hat e_2,\hat m_{\rm auto},\log\hat r)$, and `sbsi.score_inference.OutputCut` builds the
selection $W(\hat{\mathbf{x}})$ against the checkpoint's own target names, so a cut naming a column
the flow does not predict raises rather than silently producing a $P_{\rm pass}$ that does not exist.
The same object is applied by the score pass and by the population block, in NumPy and in torch
respectively, because a divergence between those two would leave $\Pi$ describing a different sample
from the one being scored with nothing to catch it. First runs are recorded in `doc/WORKLOG.md`
(cont.180) and the production numbers in cont.181. A measured-magnitude cut is nearly inert
($\Pi$ varies by 1.0% across the whole shape grid, $\iota/\mathcal I=0.0013$, uncorrected bias
$-0.28\%$); a measured-size cut is the strong case ($\Pi$ from 0.61 to 0.99, uncorrected bias
$+16.0\%$, corrected to $+0.024\%\pm0.110\%$ by the full (5.3)).

**Detection is not this.** Undetected objects have no $\hat{\mathbf{x}}$, so one cannot integrate the
flow over a region of output space that does not exist. Detection requires a separate
$P(\text{det}\mid\mathbf{x},\mathbf{n})$; see §5B.1(iii) for where it enters.

---

## 5. Computing it

Two jobs, needing different information. They evaluate the **same** derivative:

$$\underbrace{\partial_\gamma\mathbb E_\gamma[f]}_{\textbf{transport}:\ \text{average over KNOWN }\mathbf{x}}
=\underbrace{\mathrm{Cov}_0(f,s),\quad s=\mathbb E_{p(\mathbf{x},\mathbf{n}\mid\hat{\mathbf{x}})}[u]}_{\textbf{posterior}:\ \text{needs only }\hat{\mathbf{x}}}$$

§5A (transport) is **calibration** and requires $\mathbf{x}$, so it runs on simulations only.
§5B (posterior) is **inference** and is the only route that runs on real data. Fisher's identity
(2.2) is what makes the second possible: it converts a derivative of a *population* property into a
per-object *posterior expectation* of an analytic function. Transport is the degenerate case where
$\mathbf{x}$ is known, the posterior collapses to a delta, and the average becomes empirical.

§5C is not a third job: it is §5B rewritten so that $\gamma$ acts on the prior *samples* instead of on
the prior *density*. Same estimator, same answer, weaker requirements — and it is where an external
$R_{\rm blend}$ belongs if the flow cannot generate one.

### 5A. Transport — calibration (simulations only)

#### 5A.1 Recipe

$\mathrm{Cov}_0(\hat eW,s)=\partial_\gamma\mathbb E_\gamma[\hat eW]$, and the right side is evaluated
by **moving samples, not differentiating a density**, so $\nabla\log p_0$ is never needed: the
catalogue's true properties are an empirical draw from $p_0$.

```
1. draw (x, n) from the simulation catalogue          # empirical sample of p_0
2. apply S_{+-gamma} ANALYTICALLY to truth:
       primary and neighbour shapes   -> Mobius
       log T                          -> += 2 e . gamma
       separation vectors             -> sheared
3. push both legs through the flow's conditioners,
   sampling with COMMON RANDOM NUMBERS               # same latents both legs
4. apply W to the flow's OUTPUTS
5. R_S(model) = [ <e_hat>_S(+gamma) - <e_hat>_S(-gamma) ] / 2 gamma
6. compare with R_S(sim) from the paired legs;  m = R_sim / R_model - 1
```

#### 5A.2 Boundary diagnostic

(4.5) factorizes $R_{\rm sel}$ into $p_c/P_{\rm pass}$ and
$\langle\hat e\,\partial_\gamma\hat T\rangle$ at the boundary. The kept fraction tests the first
factor; the second is tested directly by restricting to a thin shell $|\hat T-T_c|<\delta$ and
comparing

$$\big\langle\hat e\,(\hat T_+-\hat T_-)/2\gamma\big\rangle_{\rm model}
\quad\text{versus}\quad
\big\langle\hat e\,(\hat T_+-\hat T_-)/2\gamma\big\rangle_{\rm truth},$$

the model side from the flow's $\pm$ legs with common random numbers, the truth side from
both-detected matched pairs carrying per-leg measured mag and size. This isolates the boundary
correlation from the response, the density and the kept fraction.

#### 5A.3 A mean-response constraint cannot fix a second moment

For a conditional-mean flow with mean head $\mu$, write
$c(\mathbf{x})\equiv\partial\mu_{\hat T}/\partial\log T$. Then

$$\partial_\gamma\hat T=c(\mathbf{x})\cdot2e
+\frac{\partial\mu_{\hat T}}{\partial\varepsilon}\cdot v_\varepsilon
+\text{residual reshaping}
\qquad\Longrightarrow\qquad
\big\langle\hat e\,\partial_\gamma\hat T\big\rangle\approx2\big\langle\hat e\,e\,c(\mathbf{x})\big\rangle.\tag{5.1}$$

A response pin that supervises the **population mean** $\bar c$ constrains $\langle c\rangle$.
Selection (4.5) needs $c$ weighted by $\hat e\,e$ **at the boundary**. If $c$ varies across the
population then

$$\big\langle\hat e\,e\,c(\mathbf{x})\big\rangle\;\neq\;\bar c\,\big\langle\hat e\,e\big\rangle,$$

so matching the first moment leaves the second unconstrained. Diagnostic: compare $c(\mathbf{x})$ as
a function of measured size — model side by autograd through the mean head, truth side from matched
pairs binned by size.

#### 5A.4 The location-family shortcut does not extend to size

If the residual flow is blind to shape (shape dropped from its conditioning) while the mean head
carries it, then $p(\hat{\mathbf{x}}\mid c)=p_{\rm resid}(\hat{\mathbf{x}}-\mu(c))$ is a location
family in the shape direction:

- **shape axis** — shearing the true shape moves only $\mu$; the grid is exact and needs no
  resampling;
- **size axis** — the residual flow *does* see size, so shearing true size reshapes the whole
  conditional density, not just its mean. **No $\mu$-only shortcut**; `log_prob` must be re-evaluated.

### 5B. Posterior — inference on real data

#### 5B.1 The algorithm

The only inputs are the measured $\hat{\mathbf{x}}_i$ and the cut $S$.

**(i) Neighbours are latent, not data.** The flow is $p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$
with $\mathbf{n}$ the *true* neighbour configuration. Real data has no such thing, so $\mathbf{n}$ is
latent exactly like $\mathbf{x}$ and must be **marginalized over a joint scene prior**
$p_0(\mathbf{x},\mathbf{n})$, never substituted from measured neighbour quantities. Consequences: the
node bank no longer depends on the galaxy and is **shared across the catalogue**; no
errors-in-variables arises on the neighbour side; but $p_0$ must encode *clustering*, not merely a
galaxy population.

**(ii) A cut on measured quantities cancels from the per-object posterior.** The cut is a
deterministic function of $\hat{\mathbf{x}}$, which is observed and passed, so

$$p(\mathbf{x},\mathbf{n}\mid\hat{\mathbf{x}},\text{passed})\propto
p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,\underbrace{\mathbb 1[\hat{\mathbf{x}}\in S]}_{=\,1}\,p_0
=p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,p_0.\tag{5.2}$$

The per-object posterior under a cut is **identical** to the one without it — §4.1 from the
algorithm's side. $P_{\rm pass}$ therefore belongs **only in the population normalization**.

**(iii) Detection does not cancel.** Detection is not determined by $\hat{\mathbf{x}}$, so
$P(\text{det}\mid\mathbf{x},\mathbf{n})$ survives in the per-object weights *and* in the
normalization. General rule: **a selection effect drops out of the per-object posterior if and only
if it is determined by data already held.**

```
node bank -- built ONCE, shared across all galaxies:

    (x_k, n_k) ~ p_0(x, n)                      # joint SCENE prior; neighbours latent
    u_k    = -( v . grad log p_0 + div v )(x_k, n_k)     # Eq 2.7, WHOLE scene
    dU_k   = d/dgamma u_k                       # for Louis, Eq 2.5
    Pdet_k = P( detected | x_k, n_k )           # classifier; does NOT cancel
    Ppass_k= Integral_S p_flow( x_hat | x_k, n_k ) d x_hat      # Eq 4.8

per galaxy i -- the only galaxy-specific quantity is L:

    L_k    = p_flow( x_hat_i | x_k, n_k )       # one flow log_prob call per node
    w_k    propto L_k * Pdet_k                     # NO cut factor -- it cancels, Eq 5.2
    s_i    = sum_k w_k u_k                                      # Eq 2.2
    I_i    = -sum_k w_k dU_k - Var_w(u)                         # Eq 2.5

population -- the SAME node bank, reweighted by Pi_k = Ppass_k * Pdet_k
             (normalized), and the SAME two formulas as the per-galaxy block:

    <s>_sel = sum_k Pi_k u_k                                    # cf. s_i
    I_sel   = -sum_k Pi_k dU_k - Var_Pi(u)                      # cf. I_i,  Eq 5.3b
    ghat    = ( sum_i s_i - N <s>_sel ) / ( sum_i I_i - N I_sel )
```

In symbols, with $\mathbb E_q$ the average under a normalized weight set $q$,

$$s_i=\mathbb E_{w_i}[u],\qquad
\mathcal I_i=-\mathbb E_{w_i}[\partial_\gamma u]-\mathrm{Var}_{w_i}(u),$$

$$\langle s\rangle_{\rm sel}=\mathbb E_\Pi[u],\qquad
\mathcal I_{\rm sel}=-\mathbb E_\Pi[\partial_\gamma u]-\mathrm{Var}_\Pi(u),
\qquad \Pi=P_{\rm pass}P_{\rm det},$$

$$\boxed{\;\hat\gamma=\frac{\sum_is_i-N\langle s\rangle_{\rm sel}}
{\sum_i\mathcal I_i-N\,\mathcal I_{\rm sel}}\;}\tag{5.3}$$

Numerator and denominator are the same pair of functionals — the mean of $u$, and Louis's combination
— each evaluated at the **posterior** weights $w_i$ and then again at the **prior-survival** weights
$\Pi$, and subtracted. §5B.2 derives the $\Pi$-weighted pair, and shows that under this document's own
isotropy assumption the *denominator's* correction is the one that survives.

**No true property appears anywhere.** This is the BFD estimator with a learned flow in place of an
analytic moment likelihood.

**Where $R_{\rm blend}$ enters.** With $\mathbf{n}$ latent and $S_\gamma$ acting on the whole scene,
$u_k$ carries neighbour components (separation, neighbour shape and size — all spin-2), so
$s_i=\sum_kw_ku_k$ picks up the blend response with no separate term. This is §3's non-degeneracy
condition at the level of weights: the blend contribution is $\sum_kw_ku_k^{(\rm nbr)}$ with
$w_k\propto L_k$, so if $L_k$ does not change as the neighbour's position angle moves across nodes,
the posterior over that angle equals the prior and the sum vanishes **by isotropy**.

**The neighbour channel is not one channel.** Split it by what shear actually moves,

$$u^{(\rm nbr)}=u_{\rm nbr\;shape}+u_{\rm nbr\;size}+u_{\rm pos}.$$

The first two are the Möbius map and $2e_s$ applied to the neighbour's own shape and size — the same
generators as the primary's, evaluated one object over — and they are nonzero for any non-uniform
shape and size prior. The third behaves completely differently.

Write the pair separation as $\mathbf r$ with $p_0(\mathbf r)\propto1+\xi(r)$, and let
$\Gamma$ be the (traceless, symmetric) shear matrix so that $v=\Gamma\mathbf r$. Then
$\nabla\!\cdot\!v=\mathrm{tr}\,\Gamma=0$, the divergence term of (2.7) drops, and only advection
survives:

$$u_{\rm pos}=-\,v\cdot\nabla\log p_0
=-\,r\,\big(\hat r^{\mathsf T}\Gamma\hat r\big)\,\frac{d\log\big(1+\xi(r)\big)}{dr},
\qquad \hat r^{\mathsf T}\Gamma\hat r=\gamma_1\cos2\phi+\gamma_2\sin2\phi,\tag{5.3a}$$

the spin-2 projection onto the pair position angle promised in §2.4.

**For an unclustered field the positional channel is identically zero** — node by node, not merely in
expectation. The reason is geometric rather than statistical: shear is area-preserving at
$O(\gamma)$, so a uniform Poisson neighbour field is *statistically invariant* under it, and a
distribution that does not move has no generator. Everything the positional channel contributes is
therefore sourced by departures from uniformity: the clustering slope in (5.3a), and the aperture
boundary below.

This is worth knowing before building a scene prior. Getting $\xi$ wrong at small separation *is* the
entire positional blend response, whereas getting the neighbour **shape** prior wrong corrupts a
channel that would be present even for a Poisson field. The two failure modes have different cures.

**The aperture is not closed under shear.** (2.7) is an ordinary gradient identity only if $S_\gamma$
maps the scene space to itself. At fixed multiplicity it does — shear moves objects, it does not
create or destroy them — so (2.7) holds sector by sector in the number of neighbours, and the
variable dimension of a scene is not by itself an obstacle. But a scene defined by an **aperture** is
not closed: shear carries neighbours across the edge, changing multiplicity at fixed aperture. What
is left over is a surface term in the separation channel, of the same species as the boundary terms
of §4.3 and with the same structure — a density at the edge times a spin-2 correlation. It is
negligible only if the aperture is wide enough that edge neighbours do not affect
$\hat{\mathbf{x}}$, which is an assumption about the catalogue's build radius rather than a theorem,
and one the recorded aperture sensitivity of the summed $R_{\rm blend}$ (WORKLOG cont.108) argues
against taking for granted.

#### 5B.2 Selection corrects both moments, not just the first

A kept object is not drawn from $p(\hat{\mathbf{x}}\mid\gamma)$. It is drawn from that density
reweighted by survival and renormalized (A.6),

$$p_{\rm keep}(\hat{\mathbf{x}}\mid\gamma)
=\frac{\displaystyle\int p_{\rm flow}(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,W(\hat{\mathbf{x}})\,
P_{\rm det}(\mathbf{x},\mathbf{n})\,p_\gamma\,d\mathbf{x}\,d\mathbf{n}}{P(\text{keep}\mid\gamma)},
\qquad P(\text{keep}\mid\gamma)=\mathbb E_{p_\gamma}[\Pi],$$

so **every** $\gamma$-derivative of a kept object's log-likelihood carries a matching derivative of
$-\log P(\text{keep}\mid\gamma)$. Differentiating once gives the known centering (4.9); differentiating
twice gives the correction that (2.6) needs and that (4.9) alone does not supply:

$$s^{\rm keep}_i=s_i-\langle s\rangle_{\rm sel},\qquad
\mathcal I^{\rm keep}_i=\mathcal I_i+\partial^2_\gamma\log P(\text{keep}\mid\gamma)\big|_0
\;\equiv\;\mathcal I_i-\mathcal I_{\rm sel}.$$

**$\mathcal I_{\rm sel}$ is Louis's identity again.** $P(\text{keep}\mid\gamma)=\int\Pi\,p_\gamma$ has
exactly the form of the marginal (1.1) — a $\gamma$-independent function integrated against the
sheared prior — with the flow's likelihood replaced by the survival probability $\Pi$. (2.5) therefore
applies verbatim, with the posterior replaced by the $\Pi$-weighted prior:

$$\partial^2_\gamma\log P(\text{keep}\mid\gamma)
=\mathbb E_\Pi\big[\partial_\gamma u\big]+\mathrm{Var}_\Pi(u)
\qquad\Longrightarrow\qquad
\boxed{\;\mathcal I_{\rm sel}=-\mathbb E_\Pi[\partial_\gamma u]-\mathrm{Var}_\Pi(u)\;}\tag{5.3b}$$

Nothing new is required to evaluate it: $u_k$, $\partial_\gamma u_k$ and $\Pi_k$ are already in the
node bank for the per-object pass, so (5.3b) is one extra weighted sum over $N_{\rm node}$ — cost
$O(N_{\rm node})$ against the estimator's $O(N_{\rm gal}N_{\rm node})$, i.e. free.

**Under this document's own symmetry assumptions the first correction vanishes and the second does
not.** §6 assumes an isotropic population at $\gamma=0$ and a rotationally invariant cut — the same
assumption that gives $\mathbb E_0[\hat eW]=0$ in §4.2. Shear is spin-2, so conjugating by a rotation
of angle $\alpha$ carries $S_\gamma$ to $S_{\gamma e^{2i\alpha}}$; with $p_0$ isotropic and $\Pi$
rotation-invariant, the survival probability is therefore invariant along that whole circle,

$$P(\text{keep}\mid\gamma)=P\big(\text{keep}\mid\gamma e^{2i\alpha}\big)\ \ \forall\alpha
\qquad\Longrightarrow\qquad
P(\text{keep}\mid\gamma)=P\big(|\gamma|\big),$$

and a function of the *components* $(\gamma_1,\gamma_2)$ that depends only on $|\gamma|$ must in fact
be a function of $|\gamma|^2$ — the cone $|\gamma|$ itself is not differentiable at the origin, so
twice-differentiability forces the square — and such a function has **zero gradient there**. Hence

$$\langle s\rangle_{\rm sel}=0\ \ \text{exactly},
\qquad
\mathcal I_{\rm sel}=\iota\,\mathbb 1_{2\times2}\neq0,$$

the second because the only rotation-invariant symmetric $2\times2$ tensor is the identity. So under
the stated assumptions the estimator's **entire** selection correction sits in the denominator, and it
is a single number $\iota$ rather than a matrix. Of the two corrections, the familiar first-order one
is the one that vanishes; keeping only it would be keeping only the term that does nothing.

**Measured, the exact zero is false — and that is informative.** On the certified V1 flow at
`|\hat{\mathbf x}|<0.6` (WORKLOG cont.174, 4–8M objects),

$$\langle s\rangle_{\rm sel}=(-0.00201\pm0.00055,\;+0.01039\pm0.00052),$$

i.e. $20\sigma$ from zero in the second component, growing to $+0.01935\pm0.00075$ when the cut is
tightened to $0.4$. The derivation above is not wrong; its **premise** is. The argument needs the
population isotropic and the cut rotation-invariant *as the model sees them*, and a trained flow is
not exactly equivariant — so $\langle s\rangle_{\rm sel}$ is a direct, calibrated measure of that
non-equivariance, available for free from a bank that has to be built anyway. Read it as a diagnostic,
not as noise. The *qualitative* claim survives intact: $\mathcal I_{\rm sel}$ moves $m$ by $+27.5\%$
against the numerator term's $+2.80\%$, so the denominator still carries $\sim\!91\%$ of the
correction, and keeping only the first-order term would still be keeping almost the wrong one.

That is §4 restated in the estimator's own language. An isotropic population under an isotropic cut
cannot acquire a preferred direction, so selection cannot produce an **additive** bias — only a
**multiplicative** one. $\langle s\rangle_{\rm sel}$ is the additive channel, and it is zero *to the
extent the flow is equivariant*;
$\mathcal I_{\rm sel}$ is the multiplicative channel, and it is $R_{\rm sel}$ (4.5) viewed from the
score side. Both are boundary-concentrated and both are driven by the same $\partial_\gamma\log T=2e$
of (4.6) — (4.5) through the first-order correlation $\langle\hat e\,e\rangle$ at the threshold,
$\iota$ through the second-order spread that the same displacement puts across it.

Keep $\langle s\rangle_{\rm sel}$ in (5.3) regardless: the symmetry is broken by PSF anisotropy, by any
cut that touches $\hat e$, and by position-dependent depth or masking (A.6), and then it is not zero.

**Magnitude and sign of $\iota$.** The vanishing of $\langle s\rangle_{\rm sel}$ says shear does not
*shift* the measured-size distribution; what it does at leading order is **diffuse** it. The *true*
size responds as $\partial_\gamma\log T=2e$ (§4.5), and the *measured* size inherits that only
through the dilution factor $c(\mathbf{x})\equiv\partial\mu_{\hat T}/\partial\log T$ of (5.1), so
$\partial_\gamma\log\hat T=2c\,e$ has zero mean and variance $D\equiv4\langle c^2e_1^2\rangle$ per
unit $\gamma^2$ (distortion convention). Diffusing a density by $D\gamma^2$ moves the kept fraction by
$-\tfrac12D\gamma^2\,p'(T_c)$, so for $ce$ uncorrelated with size

$$\iota\;\simeq\;\frac{4\langle c^2e_1^2\rangle}{P_{\rm pass}}\;
\frac{\partial p}{\partial\hat T}\bigg|_{\hat T=T_c}.\tag{5.3c}$$

Carrying $c$ matters here for the same reason §5A.3 gives: it is not $1$, it was measured to be
sign-flipped before the coupling pin, and it *varies*. Setting $c\equiv1$ recovers the bare
$4\langle e_1^2\rangle$ and is the noiseless-measurement limit, not the survey case.

Set beside (4.7) this is a tidy pairing: **$R_{\rm sel}$ sees the density at the cut edge, $\iota$ sees
its slope**, and both carry one factor of $\langle e^2\rangle$. So $\iota$ changes sign at the **peak**
of $p(\hat T)$ — a mild cut on the rising side keeps most of the sample and destroys information
($\iota>0$); an aggressive cut on the falling side keeps the responsive tail and *adds* information
($\iota<0$), growing like $1/P_{\rm pass}$.

**SETTLED, against the rule: do not trust (5.3c)'s sign for a strongly shape-dependent cut.** (2026-08-18: and do not use `--pi-azimuthal-average` to
decide whether the flow's anisotropy *causes* a non-zero $\langle s\rangle_{\rm sel}$. Averaging
$\Pi$ over rings forces $\langle s\rangle_{\rm sel}=0$ identically — it integrates a spin-2
generator against an isotropic weight — so the collapse is guaranteed and carries no
information about the flow. Measured, it falls from $9.8\times10^{-3}$ to $\sim10^{-7}$ and
$d(m)$ moves from $-0.583\%$ to $+3.766\%\pm0.261\%$. The useful conclusion is the opposite of
the intended one: the angular structure of $\Pi$ is load-bearing to the tune of 4.35% on
$d(m)$, so $\langle s\rangle_{\rm sel}$ must be evaluated and never assumed to vanish. §5B.2's
"the numerator correction vanishes for an isotropic cut" holds for an isotropic $\Pi$, not for
an isotropic CUT, and the two differ here: $|\hat x|<c$ is a disc, and $\Pi$ on it is not
isotropic.) The
production size-cut run (cont.181, V3, $\log\hat r\ge1.45$, keeping 71.1% of 8M objects) measures
$\mathcal I_{\rm sel}=-1.15075\pm0.00033$, i.e. $\iota/\mathcal I=-0.1635$ — negative at
$\sim3500\sigma$. That cut sits near the 29th percentile of $p(\log\hat r)$, the **rising** side,
where the rule above predicts $\iota>0$. The rule is wrong here, and the reason is the step it is
derived under: (5.3c) assumes $c\,e$ uncorrelated with size, and for this cut $\Pi$ runs from 0.61 to
0.99 across the shape grid, so the pass probability depends on the shape by 60% and that assumption
fails outright.

What did *not* fail is the operational term. Using the measured, negative $\mathcal I_{\rm sel}$ in
(5.3) takes a $+16.0\%$ selection bias to $+0.024\%\pm0.110\%$. So (5.3b) evaluated on the node bank
is correct and (5.3c) is an order-of-magnitude guide only — useful for deciding whether the term
matters at all, not for predicting its sign, and never a substitute for evaluating it. Where the
cut's $\Pi$ is nearly flat in shape the guide is not needed either: the same run's magnitude cut has
$\Pi$ varying by 1.0% and $\iota/\mathcal I=0.0013$, i.e. no selection correction worth making.

A.7 gives the analogous term in closed form for a *shift* parameter, where it is enormous: a cut at the
median destroys $2/\pi\approx64\%$ of the information. That toy overstates the lensing case exactly as
the symmetry argument predicts — a shift acts at $O(1)$, where spin-2 leaves only the
$O(\langle e^2\rangle)$ diffusion of (5.3c). Suppressed, then, but not small: with $c\approx1$ and
$\langle e_1^2\rangle\approx0.2$ in the distortion convention, $\iota/\mathcal I$ at the percent level is easy to reach,
which is orders of magnitude above a Stage-IV requirement on $m$. Use (5.3c) to decide how much it
matters; evaluate (5.3b) on the node bank for the number that goes into (5.3).

**It is also self-detecting.** §2.3 offered the agreement of $\sum_is_i^{\,2}$ and $\sum_i\mathcal I_i$
as a free consistency test. Under a cut the same test still works, because the sum-of-squares route
picks the correction up automatically from the centering while the Louis route does not:

$$\frac1N\sum_i\big(s_i-\langle s\rangle_{\rm sel}\big)^2
\;\approx\;\frac1N\sum_i\mathcal I_i\;-\;\mathcal I_{\rm sel},$$

both sides estimating the kept population's information. Run with the uncorrected denominator and the
two disagree by exactly the missing term, which is the cheapest way to confirm the implementation —
and, since $\langle s\rangle_{\rm sel}=0$ under isotropy, the centering drops out of the left side and
the test reduces to comparing $N^{-1}\sum_is_i^{\,2}$ with $N^{-1}\sum_i\mathcal I_i-\iota$.

In two components $\mathrm{Var}_\Pi(u)=\mathbb E_\Pi[uu^{\mathsf T}]-\mathbb E_\Pi[u]\,
\mathbb E_\Pi[u]^{\mathsf T}$, so $\mathcal I_{\rm sel}$ is a $2\times2$ matrix in general — reducing
to $\iota\mathbb 1$ under isotropy, which is itself a check on the node bank — and it is subtracted
from $\sum_i\mathcal I_i$ before the matrix solve of §6.

#### 5B.3 What (5.3) requires

1. **$\nabla\log p_0$ over the whole scene** — shape, size, flux, Sérsic and neighbour configuration,
   tractable enough to differentiate. An analytic isotropic shape prior with an exact Möbius pullback
   covers the shape channel only.
2. **A scene prior, hence clustering statistics.** Marginalizing $\mathbf{n}$ removes neighbour-side
   errors-in-variables, at the cost of needing pair statistics. It also discards the neighbours one
   can actually see. Three treatments, two correct:

   | | treatment | verdict |
   |---|---|---|
   | (a) | plug measured neighbour properties in as truth | **wrong** — errors-in-variables |
   | (b) | marginalize $\mathbf{n}$ over the prior | correct; discards real information |
   | (c) | neighbour measurements as *extra data*: joint scene inference $p(\hat{\mathbf{x}}_{\rm prim},\hat{\mathbf{x}}_{\rm nbr}\mid\mathbf{x},\mathbf{n})$ | correct and efficient; needs a scene-level flow |

   §5B.1 specifies (b); (c) requires the same scene-level likelihood that §3 requires for
   $R_{\rm blend}\neq0$.
3. **Conditioning on observing conditions.** A flow trained at one PSF and depth encodes those; PSF
   ellipticity propagates directly into measured shape, so $p(\hat{\mathbf{x}}\mid\cdot)$ must be
   conditioned on PSF size, PSF ellipticity and depth for the likelihood to be correct across a
   varying footprint.
4. **$P(\text{det}\mid\mathbf{x},\mathbf{n})$** and its integral against the sheared prior — i.e. over
   objects never observed.
5. **Cost, and effective sample size.** A $\gtrsim5$-dimensional per-object posterior. The node bank
   amortizes $u_k$, $\partial_\gamma u_k$, $P_{{\rm det},k}$ and $P_{{\rm pass},k}$ across the
   catalogue, leaving $N_{\rm gal}\times N_{\rm node}$ flow evaluations. The binding constraint is not
   that count but the **ESS per object**: $w_k\propto L_kP_{{\rm det},k}$ is importance sampling from
   $p_0$, and in $\gtrsim5$ dimensions the likelihood is far narrower than the prior, so a handful of
   nodes can carry all the weight. $s_i=\mathbb E_{w_i}[u]$ is a ratio of weighted sums, hence
   **biased, not merely noisy, at small ESS** — and the bias is systematic rather than zero-mean: it
   tracks each object's own posterior concentration without a sign that alternates between objects,
   so averaging over the catalogue does not remove it. $N_{\rm node}$ must be set by a measured ESS,
   not by budget; the same caveat is what forced large template banks in BFD. §5B.4 gives the
   separate — and prior condition — under which the denominator of (5.3) exists at all.

**Node-bank RESOLUTION is a separate requirement from ESS, and it bit (2026-08-18).** Item 5
above is about how many nodes carry weight for one galaxy. This is about how finely the bank
samples the plane at all, and it biases every galaxy in the same direction rather than the
tails. Measured on the closure test with the flow's own draws — where the answer is $g$ by
construction — the uncut control at $n=61$ ($G=2765$) reads $m=-0.442\%\pm0.215\%$ and at
$n=101$ ($G=7693$) reads $m=+0.221\%\pm0.216\%$ on the same objects, a paired shift of
$+0.664\%\pm0.005\%$ (148$\sigma$; the pair scores identical objects, so shape noise cancels and
the difference is 48$\times$ better determined than either side). That shift accounts for the
whole of the $-0.665\%\pm0.116\%$ baseline bias seen at production resolution.

The diagnostic that predicts it needs no flow: the Bartlett identity $\mathbb E_0[\partial_\gamma
u]+\mathrm{Var}_0(u)=0$ must hold for any normalised prior, so its residual on a given bank is
that bank's spurious information floor, and a positive floor inflates the denominator of (5.3)
and pushes $m$ negative. `scripts/check_quadrature.py` reports it; `scripts/predict_grid_floor.py`
converts it into a predicted bias at a realistic posterior width and got $+0.711\%\pm0.127\%$
for this step before the run finished. **Set the bank by that residual, not by eye**: at
$n=61$ it is $3.6\times10^{-3}$ of $\mathrm{Var}_0(u)$, at $n=101$ it is $1.6\times10^{-4}$.
Cost is linear in $G$.

Because every cut result is quoted as $d(m)$ against the uncut control, a resolution shift
common to both largely cancels — the $|\hat x|<0.6$ correction gives $-0.465\%\pm0.547\%$ at
$n=101$ against $-0.583\%\pm0.265\%$ at $n=61$ — so this changes what the control row means, not
the selection conclusions of §5B.2.

Note that the *dimensionality* of requirement 1 is set by the support of $v$, not by the dimension of
the scene. A primary-only shear moves the primary's shape and nothing else, so it needs
$\nabla\log p_0$ in the two-dimensional shape plane alone — a smooth, densely sampled marginal, not a
full population model. Shearing the whole scene extends the requirement to neighbour shape, neighbour
size and pair separation, roughly six dimensions per neighbour. The cheap version of §5B is therefore
genuinely cheap; it is the blend channel that carries the modelling cost.

A shape-only reduction is possible (analytic shape prior, location-family grid), but a shape-only
score yields the **shape channel alone** — not even the whole of $R_{\rm self}$, since the size
channel (3.3) is dropped with it — and neither blend (dropped with the neighbour channel by
construction, whatever the likelihood conditions on) nor selection (§4.3: the boundary term is built
mostly from the size channel).

#### 5B.4 Conditions for the estimator to exist

(5.3) divides by $\sum_i\mathcal I_i-N\mathcal I_{\rm sel}$, and $\mathcal I_i$ contains
$\mathrm{Var}_{w_i}(u)$. A denominator built out of a variance is only as good as that variance's
existence, and existence is not automatic. The §5C form of this same estimator was measured to fail
exactly here: its integrand $\partial_\gamma\log p_{\rm flow}$ carries a Hill tail index near $1.3$
— below $2$, so no finite second moment — and its denominator *grows* with bank size (fitted
exponent $+0.20$ against $-1$ for honest Monte Carlo) while a tame integrand on the identical weights
averages down. The weights were not at fault; the integrand was. WORKLOG cont.170 has the numbers.

The corresponding question for (5.3) has a clean answer, and this is the structural reason to prefer
it: §5B's integrand is analytic, so its tail is a property of a prior **you write down** — checkable
before any training, rather than discoverable only by measuring a network.

Carrying that check out settles it more favourably than expected. The naive worry is that
$\nabla\log p_0$ diverges at the edge of the ellipticity disc, so a prior that does not vanish there
fast enough would have no finite $\mathrm{Var}_0(u)$. That worry is **misplaced**, for a geometric
reason. The Möbius map preserves the unit disc, so the shear velocity field is *tangent to the
boundary*: its normal component vanishes like $1-|\varepsilon|^2$, exactly cancelling the divergence
of $\nabla\log p_0$. Writing $t\equiv|\varepsilon|^2$ and $\psi(t)\equiv\log p_0$, the generator is

$$u_a=e_a\Big[\,4-2\,\psi'(t)\,(1-t)\,\Big],\tag{5.3d}$$

so the prior enters **only** through the product $(1-t)\,\psi'(t)$, and

$$\mathbb E_0\big[u^2\big]<\infty\iff (1-t)\,\psi'(t)\in L^2(p_0),$$

i.e. the log-density's slope may not blow up *faster* than $1/(1-t)$. For the power-law family
$p_0\propto(1-t)^a$ that product is the **constant** $-a$, giving the bounded generator
$u_a=e_a(4+2a)$ and

$$\mathbb E_0\big[u^2\big]=4\,(a+2)\qquad\text{finite for every }a>-1,$$

the condition $a>-1$ being nothing but normalizability. The information exists for the *entire*
power-law family — including $a=0$, a prior that does not vanish at the edge at all.

This is verified numerically, not merely asserted. `scripts/diag5b_gate.py` differentiates the exact
Möbius pullback on a ray running into the edge and recovers $u_a/[e_a(4+2a)]=1.000000$ at edge
distances down to $10^{-7}$ for $a=0,1,2$. On the prior actually fitted
(`SmoothRadialPrior`, whose $\psi$ continues linearly in $t$ so that $(1-t)\psi'\to0$) it measures,
against §5C's numbers on the identical diagnostics:

| | §5C ($\partial_\gamma\log p_{\rm flow}$) | §5B ($u$) |
|---|---|---|
| Hill index of the integrand | $1.32-1.38$ | $9.4$ / $41.5$ / $544$ (top $5\%$/$1\%$/$0.2\%$) |
| integrand bounded? | no | yes, $\max|u|=12.27$ |
| denominator vs bank size | *grows*, exponent $+0.20$ | flat: $\mathrm{Var}_0(u)$ stable to $0.005\%$ over a $25\times$ node refinement |
| sensitivity to the truncation | — | $0.03\%$ over $r_{\max}=0.85\to0.995$ |

Two independent routes agree on the value: grid quadrature gives $\mathrm{Var}_0(u)=35.379$, and
$2\times10^6$ draws from the prior give $35.410$. Bartlett's $\mathbb E_0[u]=0$ holds to $10^{-15}$
and the curvature residual falls to $2\times10^{-5}$ of $\mathrm{Var}_0(u)$ under refinement.
**§5C's variance non-existence does not arise in §5B's shape channel**, and the default
$r_{\max}=0.95$ truncation — which never visits the edge — is not load-bearing.

Note carefully what this does *not* cover. It is the **shape channel only**. The disc-tangency
argument is special to the Möbius action on the unit disc; size and flux live on a half-line under a
dilation, and separation on the plane, so each needs its own edge analysis before its contribution to
$u$ is trusted. A bounded generator also says nothing about whether the *posterior weights* $w_i$
concentrate — that is a separate question about the flow, and the one §5B.3 item 5 is about.

**The shared node bank correlates the $s_i$.** Every galaxy is scored on the same
$(\mathbf{x}_k,\mathbf{n}_k)$, so bank realisation error is common-mode across the catalogue: it does
not average down with $N_{\rm gal}$, and $\sum_is_i^{\,2}$ therefore *understates* the variance of
$\hat\gamma$. Honest error bars come from re-running on independent banks, not from the Fisher form.
Two practical traps follow. Duplicate scenes inside a bank inflate the apparent ESS by exactly their
multiplicity — a live hazard for pair-annotated catalogues, where one physical scene appears once per
annotated neighbour and the primary is byte-identical across the group. And nested ladders, whose
rungs are column prefixes of one another, cannot see bank-realisation scatter at all; only disjoint
equal blocks can.

### 5C. Lagrangian form — shear the samples, not the prior

§5B needs $\nabla\log p_0$ over the whole scene, which is its most demanding requirement (§5B.3, item
1). That requirement is an artefact of the parametrization, not of the problem, and this section
removes it. It also gives an external $R_{\rm blend}$ a principled home, for the case where the flow
is geometry-blind and §3's neighbour channel is therefore identically zero.

#### 5C.1 The same model, reparametrized

Let $\mathbf{z}$ be the **intrinsic** (unsheared) scene and $\mathbf{x}=S_\gamma(\mathbf{z})$ the
sheared truth the flow sees. Substituting this into (1.1) — a change of variables, nothing more —
moves $\gamma$ out of the prior and into the likelihood:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p_{\rm flow}\big(\hat{\mathbf{x}}\mid S_\gamma(\mathbf{z})\big)\;
p_0(\mathbf{z})\,d\mathbf{z}.\tag{5.4}$$

(1.1) and (5.4) are the same integral. The Jacobian of $S_\gamma$ does not appear in (5.4) because
pushing samples carries it automatically; in (1.1) it is what becomes the $\nabla\!\cdot\!v$ term of
(2.7). The two forms are Eulerian and Lagrangian views of one shear flow, exactly as in (4.3) versus
(4.5).

#### 5C.2 The score without $\nabla\log p_0$

Differentiating (5.4) under the integral and dividing, as in §2.1,

$$s=\mathbb E_{\rm post}\big[\tilde u\big],\qquad
\tilde u(\mathbf{z};\hat{\mathbf{x}})\;\equiv\;\partial_\gamma\log\Big[
p_{\rm flow}\big(\hat{\mathbf{x}}\mid S_\gamma\mathbf{z}\big)\,
P_{\rm det}\big(S_\gamma\mathbf{z}\big)\Big]_0
=\Big[\nabla\log p_{\rm flow}(\hat{\mathbf{x}}\mid\cdot)+\nabla\log P_{\rm det}\Big]_{\mathbf{z}}
\!\cdot v(\mathbf{z}),\tag{5.5}$$

with $v$ the same velocity field of §2.4. The posterior is now over the **intrinsic** scene,

$$q_\gamma(\mathbf{z}\mid\hat{\mathbf{x}})\;\propto\;
p_{\rm flow}\big(\hat{\mathbf{x}}\mid S_\gamma\mathbf{z}\big)\,
P_{\rm det}\big(S_\gamma\mathbf{z}\big)\,p_0(\mathbf{z}),$$

and because $S_0=\mathrm{id}$ it coincides at $\gamma=0$ with §5B.1's. So drawing nodes
$\mathbf{z}_k\sim p_0$ leaves the self-normalized weights unchanged,
$w_k\propto L_kP_{{\rm det},k}$ with $L_k=p_{\rm flow}(\hat{\mathbf{x}}_i\mid\mathbf{z}_k)$; the cut
factor still cancels by (5.2). Louis (2.5) carries over verbatim with $u\to\tilde u$, since
$\tilde u$ is the $\gamma$-derivative of the complete-data log-likelihood in this parametrization.
Both (2.2) and (5.5) equal $\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)$, so they agree object
by object — an exact cross-check, and the identity relating them is the integration by parts that
produced (2.7).

**Every factor carrying $\gamma$ must be differentiated — including $P_{\rm det}$.** This is the one
trap of the reparametrization, and it is easy to walk into by transcribing (2.2). In the Eulerian
form $P_{\rm det}(\mathbf{x},\mathbf{n})$ is a *fixed* function of truth: $\gamma$ moves the prior
underneath it, so it enters the weights and nothing else. Here $\gamma$ moves the sample *through*
it, so it carries a derivative of its own. Omitting that channel is not a small error — it is the
same order as the retained one and can reverse the sign of $s_i$. The general rule: in the
Lagrangian form $\tilde u$ is the $\gamma$-derivative of the **whole** integrand except $p_0$, which
is now $\gamma$-free. $P_{\rm pass}$ is *not* among those factors — by (5.2) the cut evaluates to
$W(\hat{\mathbf{x}}_i)=1$ and cancels from the per-object posterior, reappearing only in the
population term (5.5c). Detection does not cancel; the cut does. That asymmetry is §5B.1(ii)–(iii),
and it is sharper here than in §5B because both would otherwise look like "selection functions of
truth."

Explicitly: let

$$\varphi_k(\gamma)\;\equiv\;\log\Big[p_{\rm flow}\big(\hat{\mathbf{x}}_i\mid S_\gamma\mathbf{z}_k\big)\,
P_{\rm det}\big(S_\gamma\mathbf{z}_k\big)\Big],$$

a scalar function of one variable along the shear curve through node $k$ — the complete-data
log-likelihood restricted to that curve. Then

$$\tilde u_k=\varphi_k'(0),\qquad
\partial_\gamma\tilde u_k=\varphi_k''(0),\qquad
\mathcal I_i=-\mathbb E_{w_i}\big[\varphi''\big]-\mathrm{Var}_{w_i}\big(\varphi'\big).\tag{5.5b}$$

Both derivatives live on the **same** curve, so one second-order forward pass returns the score and its
Louis partner together; a central second difference in $\gamma$ at fixed node is equally clean, for the
same common-random-numbers reason as below. This is the practical gain of the reparametrization. In the
Eulerian form $u$ and $\partial_\gamma u$ are two separately-derived analytic objects, each requiring
$\nabla\log p_0$ and $\nabla\!\cdot\!v$; here they are the first two Taylor coefficients of one scalar.

| | Eulerian, (2.7) | Lagrangian, (5.5) |
|---|---|---|
| $\gamma$ acts on | the prior density | the flow's **and the classifier's** inputs |
| requires | $\nabla\log p_0$ and $\nabla\!\cdot\!v$ over the whole scene | $\nabla\log p_{\rm flow}$ *and* $\nabla\log P_{\rm det}$ — autograd |
| $P_{\rm det}$ is | a fixed weight, not differentiated | a $\gamma$-carrying factor, differentiated |
| the prior must be | a differentiable density | a sample generator |
| Jacobian of $S_\gamma$ | explicit, the $\nabla\!\cdot\!v$ term | automatic |
| amortization | $u_k$ precomputed once per node | $\tilde u$ depends on $\hat{\mathbf{x}}_i$, so per (object, node) |

**The selection terms must be reparametrized too.** Both $\langle s\rangle_{\rm sel}$ and
$\mathcal I_{\rm sel}$ are written in §5B as $\Pi$-weighted moments of $u$, so taking (5.5) for the
score and leaving them alone would reintroduce $\nabla\log p_0$ through the back door and defeat the
point of this section. They do not need it either. Both are derivatives of the single scalar

$$P(\text{keep}\mid\gamma)=\mathbb E_{p_0(\mathbf{z})}\big[\Pi\big(S_\gamma\mathbf{z}\big)\big],
\qquad
\langle s\rangle_{\rm sel}=\partial_\gamma\log P\big|_0,\qquad
\mathcal I_{\rm sel}=-\partial^2_\gamma\log P\big|_0,\tag{5.5c}$$

evaluated by pushing the **same node bank** through $S_\gamma$ and asking the detection classifier and
$P_{\rm pass}$ for their values on the sheared nodes. Two JVPs through $\Pi$ suffice; because the
nodes are held fixed, a finite difference in $\gamma$ is also clean here (common random numbers) in a
way it is not on the data side. The Eulerian and Lagrangian forms agree by the same integration by
parts as before — $\mathbb E_{p_0}[\Pi u]=\mathbb E_{p_0}[\nabla\Pi\cdot v]$ — which is a useful check
on the classifier's gradients.

**This does not remove prior dependence.** $p_0$ still sets the posterior weights and the answer still
depends on it; §6's bullet stands. What is removed is the requirement that $p_0$ be available in
differentiable closed form — a modelling obstacle, not a statistical one. The cost is the last row of
the table: $\tilde u$ cannot be precomputed in the node bank the way $u_k$ can. In practice (5.5) is a
directional derivative along $v$, so a forward-mode JVP returns $\log p_{\rm flow}$ and $\tilde u$
together at roughly twice the cost of the §5B.1 evaluation, not the cost of a full gradient.

#### 5C.3 Injecting an external response

If the flow is geometry-blind, $R_{\rm blend}$ must come from outside (§3). Rather than dividing it
in afterwards, put it in the likelihood, as a shift of the measured shape proportional to the
shear-induced change in the true shape:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p_{\rm flow}
\big(\hat{\mathbf{x}}-R_b(\theta_b)\,\Delta e\;\big|\;S_\gamma\mathbf{z}\big)\,p_0(\mathbf{z})\,d\mathbf{z},
\qquad \Delta e=\big(S_\gamma\mathbf{z}\big)_e-\mathbf{z}_e,\tag{5.6}$$

where $\theta_b$ are the neighbour scalars the emulator is conditioned on. Since
$\Delta e=\gamma\,v_\varepsilon(\mathbf{z})+O(\gamma^2)$, the added term **vanishes identically at
$\gamma=0$**: it cannot disturb the unsheared model, and it is a pure response term by construction.
The score acquires a second channel,

$$s=\mathbb E_{\rm post}\Big[\underbrace{\nabla_{\mathbf{x}}\log p_{\rm flow}\cdot v}_{\text{as in }(5.5)}
\;-\;R_b(\theta_b)\,\underbrace{\nabla_{\hat e}\log p_{\rm flow}\cdot v_\varepsilon}_{\text{injected}}\Big].\tag{5.7}$$

**Sign and normalization check.** For a Gaussian flow of mean $\mu$ and variance $\sigma^2$,
$\nabla_{\hat e}\log p_{\rm flow}=-(\hat e-\mu)/\sigma^2$, so the injected channel contributes
$\mathrm{Cov}(\hat e,\,R_b(\hat e-\mu)v_\varepsilon/\sigma^2)=R_b\langle v_\varepsilon\rangle=R_b$,
using the isotropic average of $v_\varepsilon$ from §2.5. The response is $R_b$ with the expected
sign, and nothing was tuned to make it so.

**The curvature follows automatically — do not bolt it on.** (5.6) is a properly normalized
$\gamma$-family, so everything above applies to it unchanged: (5.6) simply redefines the curve
$\varphi_k$ of (5.5b) to

$$\varphi_k(\gamma)=\log\Big[p_{\rm flow}\Big(\hat{\mathbf{x}}_i-R_b(\theta_b)\,\Delta e(\gamma,\mathbf{z}_k)
\;\Big|\;S_\gamma\mathbf{z}_k\Big)P_{\rm det}\big(S_\gamma\mathbf{z}_k\big)\Big],\qquad
\Delta e(\gamma,\mathbf{z})=\big(S_\gamma\mathbf{z}\big)_e-\mathbf{z}_e,\tag{5.7b}$$

and $s_i,\mathcal I_i$ are then read off that curve by (5.5b) exactly as before. In
particular the injected channel contributes to $\mathcal I$ through *both* Louis terms, not only
through the mean — (5.7) is nothing but $\varphi_k'(0)$ for this curve. Computing $\mathcal I$ from the un-injected $\tilde u$ and adding a response
correction afterwards would repeat, one level down, the error §5B.2 corrects — a truncated model's
slope paired with the untruncated model's curvature.

**Why this is better than dividing.** The blend shift now sits *inside* the density, so $P_{\rm pass}$
(4.8) integrates the shifted distribution and the cut boundary sees blending. Selection and blending
compose. The additive form $m=R_{\rm sim}/(R_{\rm self}+R_{\rm blend})-1$ divides after selection has
already happened, which tacitly assumes the cut does not interact with the blend response — an
assumption with no support at a $\hat T$ boundary, where §4.4 shows the whole effect is concentrated.

#### 5C.4 What the injection does not buy

- **It is not self-calibrating.** Bartlett (2.4) holds for any properly normalized $\gamma$-family, so
  (2.6) remains unbiased *for the model as written* — including (5.6). But $R_b$ is external, so an
  error in it propagates linearly into $m$. This is §6's correct-specification bullet applied to the
  emulator; the estimator does not repair a wrong $R_b$, it faithfully reports it.
- **Double counting.** A flow conditioned on `nbr_flux_*` has already learned part of the blending
  response — the flux-dilution part. $R_b$ must be defined as the **residual the flow misses**, not
  the total per-pair response. Gold-v1's additive formula makes the same disjointness assumption;
  conditioning $R_b(\theta_b)$ on the same scalars the flow sees is what keeps the two separable.
- **Mean-only.** (5.6) shifts the mean of $\hat e$; blending also broadens it. The point estimate is
  therefore right and $\mathcal I$ is optimistic, so error bars from (2.6) are too small by whatever
  fraction of the scatter blending contributes.
- **Not a substitute for §3.** (5.6) supplies a number the model could not generate. A
  geometry-conditioned scene likelihood generates it, and then $R_b\to0$ by construction. The test of
  that transition is §3's own diagnostic: compute $\mathrm{Cov}(\hat e,s_{\rm nbr})\approx N^{-1}\sum_i
  \hat e_is_i^{(\rm nbr)}$ from the node bank and check it against the emulator's $R_{\rm blend}$.

#### 5C.5 The algorithm

The same three blocks as §5B.1, with $\nabla\log p_0$ nowhere in them: $p_0$ enters only as a
**sampler**.

```
node bank -- built ONCE, shared across all galaxies:

    z_k    ~ p_0(z)                     # INTRINSIC scene samples. Sampler only -- no density,
                                        #   no gradient, no divergence.
    Pdet_k = P( detected | z_k )        # classifier
    Ppass_k= Integral_S p_flow( xhat | z_k ) dxhat     # Eq 4.8, MC -- FIXED base draws
    Pi_k   = Ppass_k * Pdet_k

per galaxy i -- nothing amortizes; all per (object, node), see the 5C.2 table:

    phi_k(g) = log p_flow( xhat_i | S_g z_k )   # ONE scalar curve per node.
             + log Pdet( S_g z_k )              #   BOTH factors carry g here -- see 5C.2.
                                                #   NOT Ppass: the cut cancels, Eq 5.2.
    L_k      = p_flow( xhat_i | z_k )
    ut_k     = phi_k'(0)                        # Eq 5.5  -- score channel
    dUt_k    = phi_k''(0)                       # Eq 5.5b -- Louis channel, SAME curve
    w_k      propto L_k * Pdet_k                # Eq 5.2 -- no cut factor, it cancels
    s_i      = sum_k w_k ut_k
    I_i      = -sum_k w_k dUt_k - Var_w(ut)     # Eq 5.5b

population -- push the SAME node bank through S_g, re-ask Pi on the sheared nodes:
              (with an external R_blend, Ppass must be the SHIFTED one -- point 3)

    P(g)     = mean_k [ Ppass( S_g z_k ) * Pdet( S_g z_k ) ]      # Eq 5.5c
    <s>_sel  =   d/dg   log P |_0
    I_sel    = -d2/dg2  log P |_0
    ghat     = ( sum_i s_i - N <s>_sel ) / ( sum_i I_i - N I_sel )    # Eq 5.3
```

**The estimator.** The *form* of (5.3) is unchanged — only its ingredients are. With
$\mathbb E_q$ the average under a normalized weight set $q$ (§5B.1), and with the two curves

$$\varphi_k(\gamma)=\log\big[p_{\rm flow}(\hat{\mathbf{x}}_i\mid S_\gamma\mathbf{z}_k)\,
P_{\rm det}(S_\gamma\mathbf{z}_k)\big]
\ \ \text{per (object, node)},
\qquad
P(\gamma)=\mathbb E_{p_0(\mathbf{z})}\big[\Pi\big(S_\gamma\mathbf{z}\big)\big]
\ \ \text{once for the population},$$

the four ingredients are

$$s_i=\mathbb E_{w_i}\big[\varphi'\big],\qquad
\mathcal I_i=-\mathbb E_{w_i}\big[\varphi''\big]-\mathrm{Var}_{w_i}\big(\varphi'\big),
\qquad w_k\propto L_k\,P_{{\rm det},k},$$

$$\langle s\rangle_{\rm sel}=\big(\log P\big)'(0),\qquad
\mathcal I_{\rm sel}=-\big(\log P\big)''(0),$$

all derivatives at $\gamma=0$. Substituting into (5.3) and hiding nothing,

$$\boxed{\;\hat\gamma\;=\;
\frac{\displaystyle\sum_{i=1}^{N}\mathbb E_{w_i}\big[\varphi'\big]\;-\;N\,\big(\log P\big)'(0)}
{\displaystyle-\sum_{i=1}^{N}\Big(\mathbb E_{w_i}\big[\varphi''\big]
+\mathrm{Var}_{w_i}\big(\varphi'\big)\Big)\;+\;N\,\big(\log P\big)''(0)}\;}\tag{5.8}$$

**Every quantity in (5.8) is a first or second derivative of one of those two scalar curves**, and
neither curve requires $\nabla\log p_0$ — that is the entire content of §5C. The numerator is a
slope and the denominator a curvature, as in (2.6); the population terms subtract the part of each
that the cut alone would have produced. With an external $R_{\rm blend}$, replace $\varphi_k$ by
(5.7b) and $P_{\rm pass}$ by its shifted form (point 3); (5.8) itself does not change.

**A cheaper denominator.** Nothing forces the Louis form. A kept object has score
$s_i-\langle s\rangle_{\rm sel}$ and Bartlett applied to $p_{\rm keep}$ gives its information as the
variance of that same quantity, so

$$\hat\gamma=\frac{\sum_i\big(s_i-\langle s\rangle_{\rm sel}\big)}
{\sum_i\big(s_i-\langle s\rangle_{\rm sel}\big)^2}\tag{5.9}$$

is a legitimate first-order estimator needing **no second derivatives at all** — the §2.3 remark
$\hat\gamma=\sum s_i/\sum s_i^2$, correctly centred for the cut. §5B.2's consistency test is exactly
the statement that (5.9) and (5.8) share a denominator in expectation. Two cautions. Centring is not
optional: dropping $\langle s\rangle_{\rm sel}$ from *both* places, not just the numerator, leaves an
$O(1)$ bias, since a truncated sample has $\mathbb E_0[s]\neq0$. And the two denominators, though
equal *at* $\gamma=0$ by the information equality, drift apart away from it. Both are averages of
fixed functions of the data, so (2.3) gives each drift directly —
$\partial_\gamma\mathbb E_\gamma[(s^{\rm keep})^2]_0=\mu_3$, the third central moment of the
kept-sample score, and $\partial_\gamma\mathbb E_\gamma[\mathcal I^{\rm keep}]_0
=\mathrm{Cov}_0(\mathcal I^{\rm keep},s^{\rm keep})$ — so

$$m_{(5.9)}-m_{(5.8)}\;=\;-\gamma\,
\frac{\mu_3-\mathrm{Cov}_0\big(\mathcal I^{\rm keep},s^{\rm keep}\big)}{\mathcal I_{\rm keep}}
+O(\gamma^2).\tag{5.9b}$$

The covariance term is not optional: it vanishes only when $\mathcal I_i$ is the same for every
object, which is a Gaussian accident (A.7 is one). For a flow with a realistic spread of
$\mathcal I_i$ it carries a large fraction of (5.9b).

Both estimators are single Newton steps from $\gamma=0$, and (5.9b) is a step-*size* difference only:
the root is fixed by the numerator, which they share. Iterating either — re-evaluating its
ingredients at $\hat\gamma$ rather than at $0$ — drives the numerator to zero, so both converge to
the same MLE and (5.9b) disappears. Use (5.9) as a cheap cross-check on (5.8), or iterate it; do not
use it once, uniterated, when the multiplicative-bias budget is tight.

Four points decide whether it works.

1. **One curve, two derivatives.** $\tilde u$ and $\partial_\gamma\tilde u$ are $\varphi'(0)$ and
   $\varphi''(0)$ of the *same* scalar function (5.5b) — one second-order forward pass, or a central
   second difference. Never form $\nabla^2\log p_{\rm flow}$: this is a directional derivative along
   $v$, so the cost is $O(d)$, not $O(d^2)$.

2. **Common random numbers, twice.** The MC draws inside $P_{\rm pass}$ must be held fixed *across
   $\gamma$* and *across nodes*; otherwise $P(\gamma)$ is a noisy step function and its second
   derivative is meaningless. $\mathcal I_{\rm sel}$ is the term that dies first. The same applies to
   $\varphi_k$ if it is differenced rather than differentiated. Both are clean here **because the node
   is held fixed** — the structural advantage this parametrization has over the data side.

3. **Injection is a change of curve — but of *both* curves.** With an external $R_{\rm blend}$, (5.6)
   redefines $\varphi_k$ to (5.7b), and (5.8) is unchanged: no new term, no new estimator. It also
   redefines the survival curve, because $P_{\rm pass}$ must integrate the **shifted** density,

   $$P_{\rm pass}(\mathbf{z};\gamma)=\int_S p_{\rm flow}
   \big(\hat{\mathbf{x}}-R_b(\theta_b)\,\Delta e(\gamma,\mathbf{z})\mid S_\gamma\mathbf{z}\big)\,
   d\hat{\mathbf{x}}
   =\int_{S-R_b\Delta e}p_{\rm flow}\big(\hat{\mathbf{x}}'\mid S_\gamma\mathbf{z}\big)\,d\hat{\mathbf{x}}',$$

   i.e. the cut region translates relative to the density. At $\gamma=0$ this reduces to the node
   bank's stored $P_{{\rm pass},k}$ (since $\Delta e=0$), so the bank is still built once — but its
   $\gamma$-derivative differs, so $\langle s\rangle_{\rm sel}$ and $\mathcal I_{\rm sel}$ do too.
   Using the un-shifted $P_{\rm pass}$ here would silently discard the one thing §5C.3 was for: the
   cut boundary moving with the blend response. **Selection and blending compose only if both curves
   carry the shift.**

4. **Three free cross-checks.** (i) Where both are computable, the Eulerian (2.2) and Lagrangian (5.5)
   scores agree object by object. (ii) $\mathbb E_{p_0}[\Pi u]=\mathbb E_{p_0}[\nabla\Pi\cdot v]$ tests
   the classifier's gradients. (iii) §5B.2's consistency test,
   $N^{-1}\sum_i(s_i-\langle s\rangle_{\rm sel})^2\approx N^{-1}\sum_i\mathcal I_i-\mathcal I_{\rm sel}$,
   carries over verbatim.

**Cost.** The per-galaxy block is $O(N_{\rm gal}N_{\rm node})$ second-order forward passes — roughly
2–3× §5B.1, and the price of the last row of the 5C.2 table. The population block is $O(N_{\rm node})$
and free by comparison. Node-bank size is set by the effective sample size caveat of §5B.3, item 5.

---

## 6. Assumptions and conditions of validity

- **First order in $\gamma$.** (2.3), (2.6), (4.3) and (4.5) are leading order in shear. Cross-channel
  terms in (3.1) carry two powers of displacement and so sit in the discarded $O(\gamma^2)$.
- **$\gamma$ is two-component.** It is written as a scalar throughout for readability. In
  implementation $s_i$ is a 2-vector, $\mathcal I\equiv\mathbb E_0[ss^{\mathsf T}]$, $\mathcal I_i$ and
  $\mathcal I_{\rm sel}$ are $2\times2$ matrices, and (2.6) and (5.3) are matrix solves,
  $\hat\gamma=\big(\sum_i\mathcal I_i-N\mathcal I_{\rm sel}\big)^{-1}\big(\sum_i s_i-N\langle s\rangle_{\rm sel}\big)$.
- **Isotropy at $\gamma=0$, and a rotationally invariant cut** — required for $\mathbb E_0[\hat eW]=0$
  in §4.2. A cut on $\hat e_1$ alone violates it and reinstates the dropped term. The same assumption
  sets $\langle s\rangle_{\rm sel}=0$ and makes $\mathcal I_{\rm sel}$ a multiple of the identity
  (§5B.2), so violating it costs two terms, not one.
- **Selection corrects both moments of (2.6).** Centering the score without correcting the information
  is not a partial fix but an inconsistent one — the truncated model's slope over the untruncated
  model's curvature (A.6b). Under isotropy it is also the *wrong half*: §5B.2.
- **$\kappa=0$** in (4.6) and in the flux row of §2.4. Real surveys carry convergence together with
  shear, so the first-order flux magnification suppressed here is genuinely present in data.
- **Correct specification.** (2.4) returns the response *of the model*; misspecification appears as
  $m\neq0$ in exactly that amount. "Response from data" means "no calibration simulations", **not**
  "model-free" — a paired-simulation $R_{\rm sim}$ remains the test.
- **Prior dependence.** (2.7) contains $\nabla\log p_0$, so the inference is prior-dependent. The
  prior must be external to the simulation or the result is circular. Selection is a boundary effect
  (4.5), which amplifies prior sensitivity there.
- **Cuts on modelled outputs only.** (4.8) requires the cut variable to be an output of the flow.
- **Non-degeneracy for blending** — §3: the likelihood must depend on neighbour geometry. The V2
  model satisfies this (separation, pair angle, neighbour shape are all conditioned on); the
  implemented shear map does not exercise it, moving ellipticities only. §3, model status.
- **The scene space must be closed under $S_\gamma$.** True at fixed multiplicity, so (2.7) holds
  sector by sector in the neighbour count. Not true at fixed **aperture**, where shear carries
  neighbours across the edge and leaves a surface term in the separation channel (§5B.1). Assuming it
  away is an assumption about the catalogue's build radius.
- **The information must exist.** (5.3) divides by a variance, and heavy-tailed integrands can leave
  it without a finite population value — the measured failure mode of §5C. For §5B's **shape**
  channel this is now settled rather than assumed: the shear velocity is tangent to the ellipticity
  disc, so the generator (5.3d) depends on the prior only through $(1-t)\psi'(t)$ and stays bounded
  for the whole power-law family. Verified analytically and measured on the fitted prior — Hill index
  $9.4$ against §5C's $1.3$, $\mathrm{Var}_0(u)$ flat under both refinement and reach (§5B.4). The
  **size, flux and separation channels are not covered** by that argument and still need their own
  edge analysis.

---

## 7. Pointers

- Framework spec: `SBI_shear.md`. Certified numbers and model status: `Gold-V1.md`, `Gold-V2.md`,
  `Gold-V3.md`, `WORKLOG.md`.
- Code: `sbsi/posterior_shape.py` (posterior grid, shape prior with exact Möbius pullback),
  `sbsi/shear_map.py` (analytic $S_\gamma$), `sbsi/measurement_model.py`
  (`ConditionalMeanFlow`, `flow_drop_indices`), `sbsi/forward_model.py` +
  `sbsi/scene_model.py` (geometry-conditioned scene likelihood),
  `sbsi/selection_model.py` (detection classifier).

---

# Appendix A — the Bayesian background

Self-contained and non-lensing. Nothing in §§1–7 depends on it; it exists so those sections read as
consequences of standard statistics rather than as lensing folklore. Everything here is textbook
except the last table, which is the dictionary between the textbook names and ours.

## A.1 Four densities

| name | is | here |
|---|---|---|
| **likelihood** | $p(\text{data}\mid\text{unknowns})$ — how the data were made | the flow $p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ |
| **prior** | $p(\text{unknowns})$ — what the unknowns were before you looked | $p_0(\mathbf{x},\mathbf{n})$ |
| **evidence** | $p(\text{data})=\int$ likelihood $\times$ prior | (1.1) |
| **posterior** | $p(\text{unknowns}\mid\text{data})=$ likelihood $\times$ prior $/$ evidence | $p(\mathbf{x},\mathbf{n}\mid\hat{\mathbf{x}})$ |

The evidence is the integral that turns the other three into Bayes' theorem. It has two readings,
and the whole document rests on the second:

1. the **normalizing constant** of the posterior — a nuisance, the thing you divide by;
2. the **likelihood of whatever the prior depends on** — if the prior carries a parameter, the
   evidence is a function of that parameter, and a function of a parameter given fixed data is a
   likelihood.

## A.2 The parameter lives in the prior

$\gamma$ never appears in the flow. It appears only in $p_\gamma=p_0\circ S_{-\gamma}$, because
lensing acts on galaxies before the atmosphere and the detector do. So (1.1) *is* the likelihood
function for $\gamma$, with $(\mathbf{x},\mathbf{n})$ integrated out as nuisance parameters. Once
that sentence is accepted, §2 is just maximum likelihood applied to it, and the only difficulty left
is that the integral is intractable — which is what §A.5 handles.

## A.3 The score, and its three properties

For any model $p(y\mid\theta)$ the **score** is $s=\partial_\theta\log p(y\mid\theta)$: how strongly
the data you actually saw argue for raising $\theta$. Every identity in §2 is one line of calculus
from $\int p\,dy=1$.

**(i) It averages to zero.** Differentiate the normalization:

$$0=\partial_\theta\!\int p\,dy=\int p\;\partial_\theta\log p\,dy=\mathbb E_\theta[s].\tag{A.1}$$

At the true $\theta$ the data have no systematic opinion. Non-zero $\bar s$ means the assumed
$\theta$ is wrong — that is the measurement.

**(ii) Its variance is the curvature.** Differentiate (A.1) once more:

$$0=\int\big(\partial_\theta p\big)\,s+\int p\,\partial_\theta s
\quad\Longrightarrow\quad
\underbrace{\mathbb E[s^2]}_{\text{variance, by (A.1)}}=-\,\mathbb E\big[\partial_\theta s\big]
\;\equiv\;\mathcal I.\tag{A.2}$$

Two unrelated-sounding things — how much the votes scatter, and how sharply the log-likelihood
curves — are the same number. This is what lets §2 calibrate itself.

**(iii) Any average moves in proportion to its correlation with the score.** For fixed $f(y)$,

$$\partial_\theta\mathbb E_\theta[f]=\int f\,\partial_\theta p=\int f\,p\,\partial_\theta\log p
=\mathbb E_\theta[f\,s]\;\overset{\text{(A.1)}}{=}\;\mathrm{Cov}_\theta(f,s).\tag{A.3}$$

Plain words: the score labels which data are *evidence for a larger $\theta$*; a statistic responds
to $\theta$ exactly insofar as it is large on those same data. (A.3) is (2.3), and (A.2) is (A.3) at
$f=s$ — the one statistic whose response is its own scatter, so it needs no external calibration.

## A.4 Why the estimator is $s/\mathcal I$

Near the truth every log-likelihood is a parabola. Expanding about $\theta=0$ with $s$ the slope and
$\mathcal I$ the curvature from (A.2),

$$\log p(y\mid\theta)\approx\text{const}+s\,\theta-\tfrac12\mathcal I\,\theta^2
\quad\Longrightarrow\quad
\hat\theta=\frac{s}{\mathcal I}.\tag{A.4}$$

That is one Newton step from zero, and it is all (2.6) says: slope over curvature, accumulated over
objects. The same statement in the language of (A.3): $\mathbb E_\theta[s]\approx\mathcal I\theta$, so
dividing the observed $\bar s$ by $\mathcal I$ converts a score back into a parameter. Cramér–Rao
adds that no first-order estimator beats it, and that the correct weight per object is its own
$\mathcal I_i$ — noisy objects self-demote, with no tuning.

## A.5 Latent variables: Fisher's identity is the E-step

The obstacle: $p(\hat{\mathbf{x}}\mid\gamma)$ is an integral with no closed form, so
$\partial_\gamma\log$ of it cannot be written down. The escape: the *joint* $p_\gamma$ is analytic —
we know exactly how shear moves a galaxy. Differentiating the integral and dividing by it,

$$\partial_\gamma\log p(y\mid\theta)
=\frac{\int p(y\mid z)\,\partial_\theta p_\theta(z)\,dz}{p(y\mid\theta)}
=\mathbb E_{p(z\mid y)}\big[\partial_\theta\log p_\theta(z)\big].\tag{A.5}$$

*The score of the intractable marginal is the posterior average of the tractable joint score.* This
is (2.2), and it is the E-step of EM under another name. Louis (1982), eq. (2.5), is its second
derivative, with the extra $-\mathrm{Var}_{\rm post}$ term measuring how much the latent variables
still disagree after seeing the data.

The practical consequence, worth stating on its own: **the flow is never differentiated with respect
to $\gamma$.** It only supplies weights. All $\gamma$-dependence is analytic, in $u$.

## A.6 Truncation

If you only keep data in a region $S$, the density you are actually sampling is renormalized,

$$p_S(y\mid\theta)=\frac{p(y\mid\theta)\,\mathbb 1[y\in S]}{P(\text{pass}\mid\theta)},
\qquad P(\text{pass}\mid\theta)=\int_S p(y\mid\theta)\,dy,\tag{A.6}$$

so every kept object's log-likelihood carries $-\log P(\text{pass}\mid\theta)$ and every score becomes
$s-\partial_\theta\log P(\text{pass}\mid\theta)$. The subtracted piece has no $y$ in it — $y$ was
integrated away — so it is one number, repeated $N$ times. That is the $N\langle s\rangle_{\rm sel}$
of (5.3), and by (A.1) applied to $p_S$ it is exactly the mean score of the surviving objects: the
correction re-centres the score on the population you kept.

**The same term differentiated twice.** $-\log P(\text{pass}\mid\theta)$ sits in the log-likelihood, so
it contributes to the curvature as well as to the slope. Applying (A.2) to $p_S$,

$$\mathcal I_S=\mathcal I+\partial^2_\theta\log P(\text{pass}\mid\theta),\tag{A.6b}$$

which is the $-N\mathcal I_{\rm sel}$ of (5.3). There is nothing optional about it: (A.2) is what makes
$s/\mathcal I$ the right estimator, and (A.2) holds for $p_S$ only with *both* corrections applied.
Correcting the score alone is inconsistent — it uses the truncated model's slope with the untruncated
model's curvature. Keep the two levels distinct:

| | depends on | meaning |
|---|---|---|
| $P_{\rm pass}(\mathbf{x},\mathbf{n})$, (4.8) | one true scene | chance *this* galaxy's measurement lands in $S$ |
| $P(\text{pass}\mid\gamma)=\mathbb E_{p_\gamma}[P_{\rm pass}]$ | nothing, once averaged | population survival fraction; the normalizer in (A.6) |

The second is object-independent only because all objects face the same cut. Position-dependent depth
or masking gives $P_i(\text{pass}\mid\gamma)$ and the term becomes $\sum_i\langle s\rangle_{{\rm sel},i}$.

## A.7 The whole document in closed form

One toy where every quantity above is elementary. Truth $x\sim\mathcal N(0,\tau^2)$; the parameter
shifts the prior, $p_\theta(x)=\mathcal N(\theta,\tau^2)$ (the additive stand-in for $S_\gamma$);
measurement $y=x+\text{noise}$, $p(y\mid x)=\mathcal N(x,\sigma^2)$ — a one-dimensional flow.
Write $\nu^2=\tau^2+\sigma^2$.

| object | § | value here |
|---|---|---|
| evidence (1.1) | 1 | $p(y\mid\theta)=\mathcal N(\theta,\nu^2)$ |
| score (2.1) | 2 | $s=y/\nu^2$ |
| generator (2.7) | 2.4 | $u=\partial_\theta\log p_\theta(x)\big\rvert_0=x/\tau^2$ |
| Fisher's identity (2.2) | 2.1 | posterior mean is $y\tau^2/\nu^2$, so $\mathbb E_{\rm post}[u]=y/\nu^2=s$ ✓ |
| information (2.4) | 2.3 | $\mathcal I=\mathrm{Var}(s)=\nu^2/\nu^4=1/\nu^2$ |
| Louis (2.5) | 2.3 | $\underbrace{1/\tau^2}_{\text{complete}}-\underbrace{\sigma^2/\tau^2\nu^2}_{\text{missing}}=1/\nu^2$ ✓ |
| estimator (2.6) | 2.3 | $\hat\theta=\sum_i s_i/\sum_i\mathcal I_i=\bar y$ — the sample mean, as it must be |
| response (2.3), $f=y$ | 2.2 | $\mathrm{Cov}(y,s)=\nu^2/\nu^2=1=\partial_\theta\mathbb E_\theta[y]$ ✓ |

Now cut on $y>c$, the toy version of $\hat T>T_c$. Per-scene $P_{\rm pass}(x)=\bar\Phi\big((c-x)/\sigma\big)$
depends on the truth; the population factor does not:

$$P(\text{pass}\mid\theta)=\bar\Phi\!\left(\frac{c-\theta}{\nu}\right),\qquad
\langle s\rangle_{\rm sel}=\partial_\theta\log P(\text{pass}\mid\theta)\big\rvert_0
=\frac{\phi(c/\nu)}{\nu\,\bar\Phi(c/\nu)}
=\mathbb E\big[s\mid y>c\big],\tag{A.7}$$

the last equality because $\mathbb E[y\mid y>c]=\nu\,\phi/\bar\Phi$. So the correction in (5.3) is
literally the average score of the objects that survived, and the naive estimator that omits it
returns the truncated mean $\mathbb E[y\mid y>c]>0$ at $\theta=0$ — a selection bias, in the smallest
possible model.

Differentiating once more gives the denominator's correction (5.3b) in closed form. Write $a=c/\nu$
and $\lambda(a)=\phi(a)/\bar\Phi(a)$ for the inverse Mills ratio:

$$\mathcal I_{\rm sel}=-\partial^2_\theta\log P(\text{pass}\mid\theta)\big\rvert_0
=\frac{\lambda(\lambda-a)}{\nu^2},
\qquad
\mathcal I_S=\mathcal I-\mathcal I_{\rm sel}=\frac{1-\lambda(\lambda-a)}{\nu^2}
=\frac{\mathrm{Var}[y\mid y>c]}{\nu^4}\ \checkmark\tag{A.7b}$$

— the truncated normal's own information, as it must be. Two independent checks that (5.3b) is right:
the last equality, and evaluating $\mathcal I_{\rm sel}$ instead as
$-\mathbb E_\Pi[\partial_\theta u]-\mathrm{Var}_\Pi(u)=1/\tau^2-\mathrm{Var}[x\mid y>c]/\tau^4$, which
returns the same $\lambda(\lambda-a)/\nu^2$ after the posterior algebra. At $c=0$ — a cut at the median
— $\lambda=\sqrt{2/\pi}$ and $a=0$, so $\mathcal I_{\rm sel}/\mathcal I=2/\pi\approx0.64$: the cut
destroys nearly two thirds of the information, and an estimator that centres the score but leaves the
denominator alone reports $m=-64\%$. Nothing about that is a small correction.

Everything in §4 is this, with $y\to\hat T$ and a spin-2 shear in place of a shift — and that
replacement is not cosmetic. A shift has no orientation to average over, which is why
$\langle s\rangle_{\rm sel}\neq0$ here; for spin-2 shear the same average kills it and leaves
$\mathcal I_{\rm sel}$ as the only surviving selection term (§5B.2).

## A.8 Dictionary

| statistics | this document | weak lensing |
|---|---|---|
| score $\partial_\theta\log p$ | $s$, (2.1) | — |
| complete-data score | generator $u$, (2.7) | analytic shear velocity field |
| Fisher information | $\mathcal I$, (2.4) | inverse squared shear error |
| observed information | $\mathcal I_i$, (2.5) | per-object weight |
| $\mathrm{Cov}(f,s)$, (A.3) | (2.3) | **response** $R$; BJ02 responsivity, metacal $R$ |
| truncated likelihood, (A.6) | (5.3) | selection bias / selection response |
| E-step of EM | Fisher's identity, (2.2) | — |
| marginal likelihood | evidence, (1.1) | BFD's per-object likelihood |

## A.9 If you want the textbook version

- **Score, information, Cramér–Rao:** any mathematical-statistics text; Cox & Hinkley, *Theoretical
  Statistics*, ch. 2, is the compact one.
- **Latent variables, EM, Louis:** Dempster, Laird & Rubin (1977); Louis (1982) for the information.
- **Likelihood-ratio / score-function derivative (A.3):** Glynn (1990); the same result is REINFORCE
  in Williams (1992) and the "policy gradient" of reinforcement learning.
- **This machinery already applied to shear:** BFD, Bernstein & Armstrong (2014) — (1.1) with an
  analytic moment likelihood instead of a flow.

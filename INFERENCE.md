# INFERENCE.md — shear, response, and selection from the forward model

Derivations and algorithms only. Measured results, model status and project history live in
`WORKLOG.md`, `Gold-v1.md` and `Gold-V2.md`; the framework spec is `SBI_shear.md`.

**Summary.** The shear response is a **covariance with the score**. Selection bias is the
**size channel of that same score**, amplified by the cut into a boundary term. The blend response is
its **neighbour channel**. All three follow from one identity, and none requires a separate model.

New to the score/information machinery? **Appendix A** derives it from scratch on a one-dimensional
Gaussian, with a dictionary between the statistics names and the lensing ones. §§1–7 do not depend
on it.

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

### 4.3 Selection response = the size/flux channel

Substituting (3.1) into (4.3),

$$R_S\,\mathbb E[W]=\mathbb E[\hat eWs_{\rm shape}]+\mathbb E[\hat eWs_{\rm size}]
+\mathbb E[\hat eWs_{\rm flux}]+\mathbb E[\hat eWs_{\rm nbr}].\tag{4.4}$$

With $W\equiv1$ the flux term is zero at $\kappa=0$ and the size term is the small population-averaged
fidelity gradient (3.3). A cut changes the size term's *character*: $W$ correlates $\hat e$ with size
at the threshold, converting a gradient averaged over the whole population into a **boundary** term
(§4.4) that can dominate. **That amplified size term is the selection response** — not new physics and
not a new model, but the size channel of the same score, brought to the surface by the cut.

### 4.4 Boundary form

Following objects rather than the density,
$\partial_\gamma W=\nabla_{\hat{\mathbf{x}}}W\cdot\partial_\gamma\hat{\mathbf{x}}$, and for a hard
threshold $\nabla_{\hat T}W=\delta(\hat T-T_c)$:

$$\boxed{\;R_{\rm sel}=\frac{p_c}{P_{\rm pass}}\;
\big\langle\,\hat e\;\partial_\gamma\hat T\,\big\rangle_{\hat T=T_c}\;},
\qquad p_c\equiv p(\hat T=T_c)\tag{4.5}$$

$R_{\rm sel}$ and $R_S$ are **not** the same object: $R_S$ (4.3) is the total response of the selected
sample, while $R_{\rm sel}$ is only the piece the cut creates. In the language of (4.4),

$$R_S=R_{\rm self}\big|_S+R_{\rm blend}\big|_S+R_{\rm sel},$$

where $\big|_S$ restricts a channel to the survivors. Note also that (4.3) was derived by holding $f$
fixed and moving the density, whereas (4.5) follows objects as they cross the threshold; the two
routes are the Eulerian and Lagrangian forms of one derivative and agree at first order.

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

— the score-space analogue of the Sheldon–Huff selection response, which is stated in $\hat e$.

**Requirement.** (4.8) exists only if the flow **outputs** the cut variables. A model that takes
measured mag/size as *inputs* and outputs shape alone has no $P_{\rm pass}$: its cut variables never
move with shear. A 4D output $(\hat e_1,\hat e_2,\hat m,\hat T)$ makes them endogenous.

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

population:

    <s>_sel = sum_k Ppass_k Pdet_k u_k / sum_k Ppass_k Pdet_k
    ghat    = ( sum_i s_i - N <s>_sel ) / sum_i I_i
```

In symbols,

$$s_i=\sum_k w_ku_k,\qquad
\mathcal I_i=-\sum_kw_k\partial_\gamma u_k-\mathrm{Var}_w(u),$$

$$\boxed{\;\hat\gamma=\frac{\sum_is_i-N\langle s\rangle_{\rm sel}}{\sum_i\mathcal I_i}\;},\qquad
\langle s\rangle_{\rm sel}=\partial_\gamma\log P(\text{pass}\mid\gamma)
=\frac{\mathbb E_{p_0}[\Pi u]}{\mathbb E_{p_0}[\Pi]},\qquad \Pi=P_{\rm pass}P_{\rm det}.\tag{5.3}$$

**No true property appears anywhere.** This is the BFD estimator with a learned flow in place of an
analytic moment likelihood.

**Where $R_{\rm blend}$ enters.** With $\mathbf{n}$ latent and $S_\gamma$ acting on the whole scene,
$u_k$ carries neighbour components (separation, neighbour shape and size — all spin-2), so
$s_i=\sum_kw_ku_k$ picks up the blend response with no separate term. This is §3's non-degeneracy
condition at the level of weights: the blend contribution is $\sum_kw_ku_k^{(\rm nbr)}$ with
$w_k\propto L_k$, so if $L_k$ does not change as the neighbour's position angle moves across nodes,
the posterior over that angle equals the prior and the sum vanishes **by isotropy**.

#### 5B.2 What (5.3) requires

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
5. **Cost** — a $\gtrsim5$-dimensional per-object posterior. The node bank amortizes $u_k$,
   $\partial_\gamma u_k$, $P_{{\rm det},k}$ and $P_{{\rm pass},k}$ across the catalogue, leaving
   $N_{\rm gal}\times N_{\rm node}$ flow evaluations.

A shape-only reduction is possible (analytic shape prior, location-family grid), but a shape-only
score yields the **shape channel alone** — not even the whole of $R_{\rm self}$, since the size
channel (3.3) is dropped with it — and neither blend (§3, geometry-blind ⇒ identically zero) nor
selection (§4.3, that is the size channel).

#### 5B.3 What (5.3) weights — it is not the transport ratio

(5.3) and the calibration ratio $m$ of §5A are **different functionals of the same per-object
responses**, and the difference is not small. Suppose the flow's residual density is blind to the
true shape, so that $\log p(\hat{\mathbf{x}}\mid e)=\ell(\hat{\mathbf{x}}-\mu(e))$ is a location
family — the structure §5A.4 relies on. Write $a_i=\partial\mu/\partial\gamma$ for the response the
model assigns to object $i$. Then differentiating $s$ once through the location argument gives

$$\frac{\partial s}{\partial\hat{\mathbf{x}}}=\frac{\mathcal I_i}{a_i}.\tag{5.4}$$

Now let object $i$'s measured shape actually respond with $r_i$, which the model has no way to know.
Its two legs differ by $\hat{\mathbf x}_+-\hat{\mathbf x}_-=2\gamma r_i$, so
$s_i^+-s_i^-=2\gamma\,\mathcal I_i r_i/a_i$ and (2.6) returns

$$\boxed{\;\frac{\hat\gamma}{\gamma}=\frac{\sum_i\mathcal I_i\,(r_i/a_i)}{\sum_i\mathcal I_i}\;}
\qquad\text{against}\qquad
1+m=\frac{\langle r\rangle}{\langle a\rangle}\quad\text{(§5A)}.\tag{5.5}$$

An **information-weighted mean of the per-object response ratio**, against a **ratio of population
means**. They coincide only when $r_i/a_i$ is uncorrelated with $\mathcal I_i$.

The consequence is sharp for the blend channel. Write $r_i=a_i+b_i$ with $b_i$ the response the flow
does not model. It enters (5.5) with weight $\mathcal I_i/a_i=a_iJ_i$, where $J_i$ is the residual
flow's information for a location shift — **not** with weight $1$ as in $\langle r\rangle$. That
weight is largest for bright, well-measured, isolated objects, which are exactly the objects with the
smallest $b_i$: information and blending are anticorrelated. So the score estimator suppresses a
missing blend response by roughly $\langle a\rangle\langle aJ\rangle/\langle a^2J\rangle$, which is
far below one whenever $a_i$ varies across the catalogue.

Two things this is **not**. It is not licence to drop $R_{\rm blend}$: the suppression is a statement
about *which* objects carry the sum, so a response defect concentrated in HIGH-information objects
passes through undamped, and the per-subsample biases do not vanish — they cancel. And it is not a
free lunch in precision: $\hat\gamma$ is a weighted shear, so the weights belong in the source-density
and $n(z)$ bookkeeping like any other. The honest fix is still to make $a_i=r_i$ per object, which is
what §5C.3 does.

Measured values of (5.5) and of the subsample cancellation are in `WORKLOG.md`.

### 5C. Lagrangian form — shear the samples, not the prior

§5B needs $\nabla\log p_0$ over the whole scene, which is its most demanding requirement (§5B.2, item
1). That requirement is an artefact of the parametrization, not of the problem, and this section
removes it. It also gives an external $R_{\rm blend}$ a principled home, for the case where the flow
is geometry-blind and §3's neighbour channel is therefore identically zero.

#### 5C.1 The same model, reparametrized

Let $\mathbf{z}$ be the **intrinsic** (unsheared) scene and $\mathbf{x}=S_\gamma(\mathbf{z})$ the
sheared truth the flow sees. Substituting this into (1.1) — a change of variables, nothing more —
moves $\gamma$ out of the prior and into the likelihood:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p_{\rm flow}\big(\hat{\mathbf{x}}\mid S_\gamma(\mathbf{z})\big)\;
p_0(\mathbf{z})\,d\mathbf{z}.\tag{5.6}$$

(1.1) and (5.6) are the same integral. The Jacobian of $S_\gamma$ does not appear in (5.6) because
pushing samples carries it automatically; in (1.1) it is what becomes the $\nabla\!\cdot\!v$ term of
(2.7). The two forms are Eulerian and Lagrangian views of one shear flow, exactly as in (4.3) versus
(4.5).

#### 5C.2 The score without $\nabla\log p_0$

Differentiating (5.6) under the integral and dividing, as in §2.1,

$$s=\mathbb E_{\rm post}\big[\tilde u\big],\qquad
\tilde u(\mathbf{z};\hat{\mathbf{x}})\;\equiv\;\partial_\gamma\log p_{\rm flow}
\big(\hat{\mathbf{x}}\mid S_\gamma\mathbf{z}\big)\Big|_0
=\nabla_{\mathbf{x}}\log p_{\rm flow}(\hat{\mathbf{x}}\mid\mathbf{x})\big|_{\mathbf{z}}\cdot v(\mathbf{z}),\tag{5.7}$$

with $v$ the same velocity field of §2.4 and the posterior weights unchanged from §5B.1,
$w_k\propto L_kP_{{\rm det},k}$. Louis (2.5) carries over verbatim with $u\to\tilde u$, since
$\tilde u$ is the $\gamma$-derivative of the complete-data log-likelihood in this parametrization.
Both (2.2) and (5.7) equal $\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)$, so they agree object
by object — an exact cross-check, and the identity relating them is the integration by parts that
produced (2.7).

| | Eulerian, (2.7) | Lagrangian, (5.7) |
|---|---|---|
| $\gamma$ acts on | the prior density | the flow's conditioning inputs |
| requires | $\nabla\log p_0$ and $\nabla\!\cdot\!v$ over the whole scene | $\nabla_{\mathbf{x}}\log p_{\rm flow}$ — autograd |
| the prior must be | a differentiable density | a sample generator |
| Jacobian of $S_\gamma$ | explicit, the $\nabla\!\cdot\!v$ term | automatic |
| amortization | $u_k$ precomputed once per node | $\tilde u$ depends on $\hat{\mathbf{x}}_i$, so per (object, node) |

**This does not remove prior dependence.** $p_0$ still sets the posterior weights and the answer still
depends on it; §6's bullet stands. What is removed is the requirement that $p_0$ be available in
differentiable closed form — a modelling obstacle, not a statistical one. The cost is the last row:
$\tilde u$ cannot be precomputed in the node bank the way $u_k$ can. In practice (5.7) is a
directional derivative along $v$, so a forward-mode JVP returns $\log p_{\rm flow}$ and $\tilde u$
together at roughly twice the cost of the §5B.1 evaluation, not the cost of a full gradient.

#### 5C.3 Injecting an external response

If the flow is geometry-blind, $R_{\rm blend}$ must come from outside (§3). Rather than dividing it
in afterwards, put it in the likelihood, as a shift of the measured shape proportional to the
shear-induced change in the true shape:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p_{\rm flow}
\big(\hat{\mathbf{x}}-R_b(\theta_b)\,\Delta e\;\big|\;S_\gamma\mathbf{z}\big)\,p_0(\mathbf{z})\,d\mathbf{z},
\qquad \Delta e=\big(S_\gamma\mathbf{z}\big)_e-\mathbf{z}_e,\tag{5.8}$$

where $\theta_b$ are the neighbour scalars the emulator is conditioned on. Since
$\Delta e=\gamma\,v_\varepsilon(\mathbf{z})+O(\gamma^2)$, the added term **vanishes identically at
$\gamma=0$**: it cannot disturb the unsheared model, and it is a pure response term by construction.
The score acquires a second channel,

$$s=\mathbb E_{\rm post}\Big[\underbrace{\nabla_{\mathbf{x}}\log p_{\rm flow}\cdot v}_{\text{as in }(5.7)}
\;-\;R_b(\theta_b)\,\underbrace{\nabla_{\hat e}\log p_{\rm flow}\cdot v_\varepsilon}_{\text{injected}}\Big].\tag{5.9}$$

**Sign and normalization check.** For a Gaussian flow of mean $\mu$ and variance $\sigma^2$,
$\nabla_{\hat e}\log p_{\rm flow}=-(\hat e-\mu)/\sigma^2$, so the injected channel contributes
$\mathrm{Cov}(\hat e,\,R_b(\hat e-\mu)v_\varepsilon/\sigma^2)=R_b\langle v_\varepsilon\rangle=R_b$,
using the isotropic average of $v_\varepsilon$ from §2.5. The response is $R_b$ with the expected
sign, and nothing was tuned to make it so.

**Why this is better than dividing.** The blend shift now sits *inside* the density, so $P_{\rm pass}$
(4.8) integrates the shifted distribution and the cut boundary sees blending. Selection and blending
compose. The additive form $m=R_{\rm sim}/(R_{\rm self}+R_{\rm blend})-1$ divides after selection has
already happened, which tacitly assumes the cut does not interact with the blend response — an
assumption with no support at a $\hat T$ boundary, where §4.4 shows the whole effect is concentrated.

#### 5C.4 What the injection does not buy

- **It is not self-calibrating.** Bartlett (2.4) holds for any properly normalized $\gamma$-family, so
  (2.6) remains unbiased *for the model as written* — including (5.8). But $R_b$ is external, so an
  error in it propagates linearly into $m$. This is §6's correct-specification bullet applied to the
  emulator; the estimator does not repair a wrong $R_b$, it faithfully reports it.
- **Double counting.** A flow conditioned on `nbr_flux_*` has already learned part of the blending
  response — the flux-dilution part. $R_b$ must be defined as the **residual the flow misses**, not
  the total per-pair response. Gold-v1's additive formula makes the same disjointness assumption;
  conditioning $R_b(\theta_b)$ on the same scalars the flow sees is what keeps the two separable.
- **Mean-only.** (5.8) shifts the mean of $\hat e$; blending also broadens it. The point estimate is
  therefore right and $\mathcal I$ is optimistic, so error bars from (2.6) are too small by whatever
  fraction of the scatter blending contributes.
- **Not a substitute for §3.** (5.8) supplies a number the model could not generate. A
  geometry-conditioned scene likelihood generates it, and then $R_b\to0$ by construction. The test of
  that transition is §3's own diagnostic: compute $\mathrm{Cov}(\hat e,s_{\rm nbr})\approx N^{-1}\sum_i
  \hat e_is_i^{(\rm nbr)}$ from the node bank and check it against the emulator's $R_{\rm blend}$.

---

## 6. Assumptions and conditions of validity

- **First order in $\gamma$.** (2.3), (2.6), (4.3) and (4.5) are leading order in shear. Cross-channel
  terms in (3.1) carry two powers of displacement and so sit in the discarded $O(\gamma^2)$.
- **$\gamma$ is two-component.** It is written as a scalar throughout for readability. In
  implementation $s_i$ is a 2-vector, $\mathcal I\equiv\mathbb E_0[ss^{\mathsf T}]$ and $\mathcal I_i$
  are $2\times2$ matrices, and (2.6) and (5.3) are matrix solves,
  $\hat\gamma=\big(\sum_i\mathcal I_i\big)^{-1}\sum_i s_i$.
- **Isotropy at $\gamma=0$, and a rotationally invariant cut** — required for $\mathbb E_0[\hat eW]=0$
  in §4.2. A cut on $\hat e_1$ alone violates it and reinstates the dropped term.
- **$\kappa=0$** in (4.6) and in the flux row of §2.4. Real surveys carry convergence together with
  shear, so the first-order flux magnification suppressed here is genuinely present in data.
- **Correct specification.** (2.4) returns the response *of the model*; misspecification appears as
  $m\neq0$ in exactly that amount. "Response from data" means "no calibration simulations", **not**
  "model-free" — a paired-simulation $R_{\rm sim}$ remains the test.
- **Prior dependence.** (2.7) contains $\nabla\log p_0$, so the inference is prior-dependent. The
  prior must be external to the simulation or the result is circular. Selection is a boundary effect
  (4.5), which amplifies prior sensitivity there.
- **Cuts on modelled outputs only.** (4.8) requires the cut variable to be an output of the flow.
- **Non-degeneracy for blending** — §3: the likelihood must depend on neighbour geometry.

---

## 7. Pointers

- Framework spec: `SBI_shear.md`. Certified numbers and model status: `Gold-v1.md`, `Gold-V2.md`,
  `WORKLOG.md`.
- Code: `sbs_shear/posterior_shape.py` (posterior grid, shape prior with exact Möbius pullback),
  `sbs_shear/shear_map.py` (analytic $S_\gamma$), `sbs_shear/measurement_model.py`
  (`ConditionalMeanFlow`, `flow_drop_indices`), `sbs_shear/forward_model.py` +
  `sbs_shear/scene_model.py` (geometry-conditioned scene likelihood),
  `sbs_shear/selection_model.py` (detection classifier).

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
correction re-centres the score on the population you kept. Keep the two levels distinct:

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
possible model. Everything in §4 is this, with $y\to\hat T$ and a spin-2 shear in place of a shift.

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

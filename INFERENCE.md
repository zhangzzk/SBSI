# INFERENCE.md — shear, response, and selection from the forward model

Derivations and algorithms only. Measured results, model status and project history live in
`WORKLOG.md`, `Gold-v1.md` and `Gold-V2.md`; the framework spec is `SBI_shear.md`.

**Summary.** The shear response is a **covariance with the score**. Selection bias is the
**size/flux channel of that same score**, made visible by the cut. The blend response is its
**neighbour channel**. All three follow from one identity, and none requires a separate model.

---

## 1. Setup and notation

| symbol | meaning |
|---|---|
| $\mathbf{x}$ | true properties of the primary: shape $e$, size $T$, flux, Sérsic $n$ |
| $\mathbf{n}$ | true neighbour configuration: separations, position angles, shapes, sizes, fluxes |
| $\hat{\mathbf{x}}$ | measured catalogue quantities; for a 4D flow $\hat{\mathbf{x}}=(\hat e_1,\hat e_2,\hat m,\hat s)$ |
| $p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ | the learned flow — **shear-free** |
| $p_0$ | intrinsic (unsheared) prior over the whole scene |
| $S_\gamma$ | analytic shear map, acting on **truth** |
| $W(\hat{\mathbf{x}})$ | cut indicator, e.g. $\mathbb 1[\hat s>s_c]$ |

Shear appears in exactly one place — it shifts the prior:

$$p(\hat{\mathbf{x}}\mid\gamma)=\int p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\;
p_0\big(S_{-\gamma}(\mathbf{x},\mathbf{n})\big)\,d\mathbf{x}\,d\mathbf{n}. \tag{1.1}$$

Everything below follows from this structure alone.

---

## 2. The score, and three identities

$$s(\hat{\mathbf{x}})\;\equiv\;\partial_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)\big|_{\gamma=0}. \tag{2.1}$$

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
=\mathbb E_\gamma[f\,s],$$

and since $\mathbb E_0[s]=0$,

$$\boxed{\;\partial_\gamma\mathbb E_\gamma[f]\big|_0=\mathrm{Cov}_0\big(f(\hat{\mathbf{x}}),\,s(\hat{\mathbf{x}})\big)\;}\tag{2.3}$$

(the likelihood-ratio / score-function estimator). With $f=\hat e$ this is the shear response as a
sample covariance — no paired $\pm\gamma$ renders and no finite differences.

### 2.3 Bartlett and Louis

$$\mathbb E[s]=0,\qquad \mathcal I\equiv\mathrm{Var}[s]=-\mathbb E[\partial_\gamma s],\tag{2.4}$$

$$\partial^2_\gamma\log p(\hat{\mathbf{x}}\mid\gamma)
=\mathbb E_{\rm post}\big[\partial^2_\gamma\log p_\gamma\big]
+\mathrm{Var}_{\rm post}\big[\partial_\gamma\log p_\gamma\big].\tag{2.5}$$

(2.5) reads *observed information = complete-data information $-$ missing information*. Hence

$$\hat\gamma=\frac{\sum_i s_i}{\sum_i\mathcal I_i}
\qquad\Longrightarrow\qquad
\partial_\gamma\mathbb E[\hat\gamma]=1\ \text{ by construction},\tag{2.6}$$

with no external responsivity applied afterwards and no $R_{\rm self}+R_{\rm blend}$ summed by hand.
By Cramér–Rao (2.6) is the minimum-variance first-order estimator, and the per-object weight is its
own Fisher information rather than a tuned quantity.

### 2.4 The generator

Let $v\equiv\partial_\gamma S_\gamma|_0$ be the shear velocity field on truth. Since $p_\gamma$ is the
pushforward of $p_0$ along $S_\gamma$, the continuity equation
$\partial_\gamma p_\gamma+\nabla\!\cdot\!(p_\gamma v)=0$ gives

$$\boxed{\;u(\mathbf{x},\mathbf{n})\;\equiv\;\partial_\gamma\log p_\gamma\big|_0
\;=\;-\Big(v\cdot\nabla\log p_0+\nabla\!\cdot\! v\Big)\;}\tag{2.7}$$

The channel velocities, all closed-form:

| channel | velocity $v$ | order |
|---|---|---|
| shape (reduced-shear $\varepsilon$, ngmix `g`) | $v_e=\partial_g\!\left[\dfrac{e+g}{1+\bar g e}\right]_0=\big(1-e^2,\;i(1+e^2)\big)$ for $(g_1,g_2)$ | spin-2 |
| log size (distortion $e$) | $v_{\log T}=2e$ — see §4.5 | spin-2, **no isotropic part** |
| log flux | $0$ at first order when $\kappa=0$; magnification $\mu=\big[(1-\kappa)^2-|g|^2\big]^{-1}$ otherwise | — |
| neighbour separation | $\delta r_i=\gamma_{ij}r_j$ | spin-2 in the pair position angle |
| neighbour shape | Möbius, as for the primary | spin-2 |

### 2.5 Limits

- **Noiseless measurement.** The posterior collapses to a delta, $s\to u$, and
  $\mathrm{Cov}(e,s)=\partial_\gamma\langle e\rangle$: the Bernstein & Jarvis responsivity
  $\mathcal R=2(1-e_{\rm rms}^2)$ in the distortion convention, $\mathcal R\to1$ for reduced shear
  (the isotropic average of $v_e$ above is exactly $1$).
- **Low signal-to-noise.** The posterior tends to the prior, so $s\to\mathbb E_{p_0}[u]=0$: the object
  contributes nothing rather than contributing noise.
- **No cut.** Only $s_{\rm shape}$ correlates with $\hat e$, so the response reduces to $R_{\rm self}$
  (§3).

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

$$R=\underbrace{\mathrm{Cov}(\hat e,\,s_{\rm shape})}_{R_{\rm self}}
+\underbrace{\mathrm{Cov}(\hat e,\,s_{\rm nbr})}_{R_{\rm blend}}.\tag{3.2}$$

**The two combine by covariance, not by hand-summation.** There is no separate additive blend term to
bolt on; it is the neighbour channel of the same score.

**Non-degeneracy condition.** $\mathrm{Cov}(\hat e,s_{\rm nbr})\equiv 0$ unless
$p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})$ **depends on neighbour geometry** — separation, pair
position angle, neighbour shape — the quantities shear distorts with spin 2. A likelihood conditioned
only on *scalar* neighbour fluxes is invariant under rotating a neighbour about the primary at fixed
flux, so the neighbour score is uncorrelated with $\hat e$ and the term vanishes identically. See
§5B.1 for the same statement at the level of posterior weights.

**Consequence.** Under a geometry-blind likelihood, $R_{\rm blend}$ cannot be recovered from the
model and must be supplied externally and added, i.e. $m=R_{\rm sim}/(R_{\rm flow}+R_{\rm blend})-1$.
The additive form is a symptom of the missing conditioning, not a modelling choice.

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
invariant cut (on $\hat s$, $\hat m$, S/N — **not** on $\hat e_1$ alone) the numerator vanishes,
$\mathbb E_0[\hat eW]=0$, killing the second term. What remains is (2.3) with $f=\hat eW$:

$$\boxed{\;R_S=\frac{\mathrm{Cov}_0\big(\hat e\,W(\hat{\mathbf{x}}),\;s(\hat{\mathbf{x}})\big)}{\mathbb E_0[W]}\;}\tag{4.3}$$

Selection changed only $f:\hat e\to\hat eW$; the identity did not care. $\mathbb E_0[W]$ is the kept
fraction, converting a sum over all into a mean over the selected.

### 4.3 Selection response = the size/flux channel

Substituting (3.1) into (4.3),

$$R_S\,\mathbb E[W]=\mathbb E[\hat eWs_{\rm shape}]+\mathbb E[\hat eWs_{\rm size}]
+\mathbb E[\hat eWs_{\rm flux}]+\mathbb E[\hat eWs_{\rm nbr}].\tag{4.4}$$

With $W\equiv1$ the size and flux terms vanish: $\hat e$ is uncorrelated with an isotropic dilation.
With a cut, $W$ correlates $\hat e$ with size and $\mathbb E[\hat eWs_{\rm size}]\neq0$. **That term
is the selection response** — not new physics and not a new model, but the size channel of the same
score, switched on by the cut.

### 4.4 Boundary form

Following objects rather than the density,
$\partial_\gamma W=\nabla_{\hat{\mathbf{x}}}W\cdot\partial_\gamma\hat{\mathbf{x}}$, and for a hard
threshold $\nabla_{\hat s}W=\delta(\hat s-s_c)$:

$$\boxed{\;R_{\rm sel}=\frac{p(\hat s=s_c)}{P_{\rm pass}}\;
\big\langle\,\hat e\;\partial_\gamma\hat s\,\big\rangle_{\hat s=s_c}\;}\tag{4.5}$$

The selection response is a **boundary integral**: density at the cut edge times the correlation
between measured shape and the shear response of the cut variable, evaluated on the edge. Interior
objects contribute nothing.

Any **isotropic** part of $\partial_\gamma\hat s$ contributes
$\langle\hat e\rangle_{\rm boundary}\times\langle\partial_\gamma\hat s\rangle=0$. **Only the spin-2,
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
\sim\frac{p_c}{P_{\rm pass}}\cdot2\langle e^2\rangle\,R_{\rm meas}.\tag{4.7}$$

### 4.6 Consequences for cut direction

| cut | mechanism | predicted shift |
|---|---|---|
| keep large ($\hat s>s_c$) | aligned galaxies grow ⇒ over-represented | $>0$ |
| keep bright ($\hat m<m_c$) | at $\kappa=0$ no first-order flux magnification, but a size-scaled aperture (e.g. Kron) makes $\hat m$ inherit the size response ⇒ also spin-2; aligned galaxies measure brighter ⇒ kept | $>0$ |
| cut on a shear-invariant true property | $\partial_\gamma\hat s=0$ | $0$ exactly |

### 4.7 The selection function, and the 4D-output requirement

The cut is a function of the flow's **own output variables**, so the selection function is *derived*,
not learned:

$$P_{\rm pass}(\mathbf{x},\mathbf{n})=\int_S p(\hat{\mathbf{x}}\mid\mathbf{x},\mathbf{n})\,d\hat{\mathbf{x}}
\tag{4.8}$$

— Monte Carlo: draw $n_{\rm samples}$ from the flow, count the fraction inside $S$. It enters the
marginal likelihood's normalization, and in score form is a single centering,

$$s_{\rm sel}=s-\langle s\rangle_{\rm selected}\tag{4.9}$$

(the Sheldon–Huff selection response).

**Requirement.** (4.8) exists only if the flow **outputs** the cut variables. A model that takes
measured mag/size as *inputs* and outputs shape alone has no $P_{\rm pass}$: its cut variables never
move with shear. A 4D output $(\hat e_1,\hat e_2,\hat m,\hat s)$ makes them endogenous.

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

### 5A.1 Transport — calibration (simulations only)

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

### 5A.2 Boundary diagnostic

(4.5) factorizes $R_{\rm sel}$ into $p_c/P_{\rm pass}$ and
$\langle\hat e\,\partial_\gamma\hat s\rangle$ at the boundary. The kept fraction tests the first
factor; the second is tested directly by restricting to a thin shell $|\hat s-s_c|<\delta$ and
comparing

$$\big\langle\hat e\,(\hat s_+-\hat s_-)/2\gamma\big\rangle_{\rm model}
\quad\text{versus}\quad
\big\langle\hat e\,(\hat s_+-\hat s_-)/2\gamma\big\rangle_{\rm truth},$$

the model side from the flow's $\pm$ legs with common random numbers, the truth side from
both-detected matched pairs carrying per-leg measured mag and size. This isolates the boundary
correlation from the response, the density and the kept fraction.

### 5A.3 A mean-response constraint cannot fix a second moment

For a conditional-mean flow with mean head $\mu$, write
$c(\mathbf{x})\equiv\partial\mu_{\hat s}/\partial\log T$. Then

$$\partial_\gamma\hat s=c(\mathbf{x})\cdot2e+\frac{\partial\mu_{\hat s}}{\partial e}\cdot v_e
+\text{residual reshaping}
\qquad\Longrightarrow\qquad
\big\langle\hat e\,\partial_\gamma\hat s\big\rangle\approx2\big\langle\hat e\,e\,c(\mathbf{x})\big\rangle.\tag{5.1}$$

A response pin that supervises the **population mean** $\bar c$ constrains $\langle c\rangle$.
Selection (4.5) needs $c$ weighted by $\hat e\,e$ **at the boundary**. If $c$ varies across the
population then

$$\big\langle\hat e\,e\,c(\mathbf{x})\big\rangle\;\neq\;\bar c\,\big\langle\hat e\,e\big\rangle,$$

so matching the first moment leaves the second unconstrained. Diagnostic: compare $c(\mathbf{x})$ as
a function of measured size — model side by autograd through the mean head, truth side from matched
pairs binned by size.

### 5A.4 The location-family shortcut does not extend to size

If the residual flow is blind to shape (shape dropped from its conditioning) while the mean head
carries it, then $p(\hat{\mathbf{x}}\mid c)=p_{\rm resid}(\hat{\mathbf{x}}-\mu(c))$ is a location
family in the shape direction:

- **shape axis** — shearing the true shape moves only $\mu$; the grid is exact and needs no
  resampling;
- **size axis** — the residual flow *does* see size, so shearing true size reshapes the whole
  conditional density, not just its mean. **No $\mu$-only shortcut**; `log_prob` must be re-evaluated.

### 5B.1 Posterior — inference on real data

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
    D_k    = P( detected | x_k, n_k )           # classifier; does NOT cancel
    Wp_k   = Integral_S p_flow( x_hat | x_k, n_k ) d x_hat      # Eq 4.8

per galaxy i -- the only galaxy-specific quantity is L:

    L_k    = p_flow( x_hat_i | x_k, n_k )       # one flow log_prob call per node
    w_k    propto L_k * D_k                     # NO cut factor -- it cancels, Eq 5.2
    s_i    = sum_k w_k u_k                                      # Eq 2.2
    I_i    = -sum_k w_k dU_k - Var_w(u)                         # Eq 2.5

population:

    <s>_sel = sum_k Wp_k D_k u_k / sum_k Wp_k D_k
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

### 5B.2 What (5.3) requires

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
   $\partial_\gamma u_k$, $D_k$ and $W\!p_k$ across the catalogue, leaving
   $N_{\rm gal}\times N_{\rm node}$ flow evaluations.

A shape-only reduction is possible (analytic shape prior, location-family grid), but a shape-only
score yields $R_{\rm self}$ alone: no blend (§3, geometry-blind ⇒ identically zero) and no selection
(§4.3, that is the size channel).

---

## 6. Assumptions and conditions of validity

- **First order in $\gamma$.** (2.3), (2.6), (4.3) and (4.5) are leading order in shear.
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

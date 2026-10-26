# MATH.md — the shear estimator by the Bartlett route

**Zekang's derivation** (draft, 2026-07-29), written out and carried to the end.

This reaches the same estimator as `INFERENCE.md` §5C by a different and shorter road. `INFERENCE.md`
gets the denominator from **Louis** — the curvature of the complete-data likelihood minus the
posterior spread. Here it comes from **Bartlett** — the score's own variance. The two denominators
are equal in expectation; §5 derives (5.8) in full. Neither §3 nor §5 needs $\nabla\log p_0$: shear
acts on the *samples*, not on the prior density.

§6 derives the other parametrization — `INFERENCE.md` §5B, equation (5.3) — in which shear acts on
the **prior** instead. It stands alone and can be read without §§1–3 or §5; it is the same estimator,
not a competing one, and its Step 8 proves so by a change of variables. It is also where
$\nabla\log p_0$ comes from. It is retained for the earlier finite-M score study; the catalogue-prior
null-closure path now follows §5 and numerically differentiates the marginalized log likelihood at
$g=0$. §7 sets the Bartlett denominator against (5.3)'s.

For the current response-regularized mean-affine flow, the matched finite-
catalogue closure verifies this equality directly: exhaustive numerical
derivatives at `h=0.00125` agree with exact Torch autograd, and both
$E[I]=\mathrm{Var}(s)$ and the additive null pass.  The 10,000-object full-
prior importance run also closes the identity and draw ladder, although its
g2 score-tail estimate remains marginal (WORKLOG cont.202).  The paired local
nonzero implementation is sampler-stable and statistically consistent with
zero multiplicative bias at about 0.5% precision, but its response-influence
Hill indices are below two and its off-diagonal responses are 10--14% at more
than four sigma.  The five-view estimator omits the mixed Hessian under an
exact rotational-symmetry assumption that the learned likelihood does not
meet (WORKLOG cont.203).

This is specifically a *local* expansion.  Full likelihood profiles of 10,000
saved mocks recover axis-aligned injected shears 0.02 and 0.05 within 0.31
profile-curvature errors in all four tests, whereas a single Newton step from
zero overshoots at 0.02 and encounters negative observed information at 0.05.
The mock mean response remains linear to about four parts in 100,000 between
those amplitudes.  Thus mean linearity does not make the catalogue-marginalized
likelihood a common Gaussian location family: changing posterior mixture
responsibilities makes its curvature shear-dependent.  Finite-shear inference
must profile or iteratively recentre; the null/local estimator still does not
establish `|m|<0.2%` (WORKLOG cont.204).

Equations are tagged `M.x` to avoid collision with `INFERENCE.md`.

---

## 1. The model

Let $\theta=(\mathbf{x},\mathbf{n})$ be the **intrinsic** scene — primary and neighbours, unsheared —
drawn from $\pi(\theta)$. Shear maps it to $S_\gamma\theta$, and the measurement and detection
machinery act on the *sheared* truth:

For the SBSI closure in this repository, $S_\gamma$ is deliberately the same
shape-only operation as the image simulation: it applies the exact
reduced-shear map to every object's intrinsic ellipticity and leaves flux,
size, position, pair separation, and neighbour membership unchanged.  There
is no magnification or positional shear in this project scope.

$$A(\hat{\mathbf{x}}\mid\gamma)\;=\;\int d\theta\;\pi(\theta)\;
L\big(\hat{\mathbf{x}}\mid S_\gamma\theta\big)\;P_{\rm det}\big(S_\gamma\theta\big),\tag{M.1}$$

with $L$ the flow and $P_{\rm det}$ the detection classifier. This is the Lagrangian form: $\gamma$
sits in the likelihood, not the prior, so $\pi$ is needed only as a **sampler**. The Jacobian of
$S_\gamma$ never appears because pushing samples carries it.

$A$ is **not normalized** — $\int A\,d\hat{\mathbf{x}}=P(\text{det}\mid\gamma)$, which itself moves
with $\gamma$. §4 supplies the normalizer; until then $A$ is a working quantity, not a density.

**Optional: an external blend response.** If $R_{\rm blend}$ comes from an emulator rather than from
the flow, inject it as a shift of the measured shape,

$$L\big(\hat{\mathbf{x}}\mid S_\gamma\theta\big)\;\longrightarrow\;
L\big(\hat{\mathbf{x}}-R_b(\theta_b)\,\Delta e\;\big|\;S_\gamma\theta\big),
\qquad \Delta e=\big(S_\gamma\theta\big)_e-\theta_e.\tag{M.2}$$

Everything below is unchanged — (M.2) only redefines the curve $\varphi_\theta$ of (M.3).
Since $\Delta e=\gamma v_\varepsilon+O(\gamma^2)$ the shift vanishes at $\gamma=0$, so it cannot
disturb the unsheared model and is a pure response term.

> **Sign.** "Measured $=$ flow output $+$ blend shift" means $\hat{\mathbf{x}}_{\rm flow}
> =\hat{\mathbf{x}}-R_b\Delta e$, hence the **minus** inside $L$. Writing $+$ flips the sign of
> $R_b$ and silently reverses the blend response.

---

## 2. The score

Differentiate (M.1) under the integral and divide by $A$ — the bracket becomes a posterior:

$$\boxed{\;\hat s(\hat{\mathbf{x}})\;=\;\partial_\gamma\log A\big|_0
\;=\;\mathbb E_{\rm post}\big[\varphi_\theta'(0)\big]\;}
\qquad
\varphi_\theta(\gamma)\equiv\log\Big[L\big(\hat{\mathbf{x}}\mid S_\gamma\theta\big)\,
P_{\rm det}\big(S_\gamma\theta\big)\Big],\tag{M.3}$$

with posterior weights $q(\theta\mid\hat{\mathbf{x}})\propto L(\hat{\mathbf{x}}\mid\theta)
P_{\rm det}(\theta)\pi(\theta)$. Drawing nodes $\theta_k\sim\pi$ makes these self-normalized weights
$w_k\propto L_kP_{{\rm det},k}$.

> **Both factors carry $\gamma$.** $\varphi_\theta$ is the log of the *whole* integrand except $\pi$.
> In the Eulerian form $P_{\rm det}(\theta)$ is a fixed function of truth — $\gamma$ moves the prior
> underneath it, so it only weights. Here $\gamma$ moves the sample *through* it, so it must be
> differentiated too. Omitting that channel is the same order as keeping it and can flip the sign of
> $\hat s$; see §8.
>
> **$P_{\rm pass}$ is not one of those factors.** The cut is a deterministic function of
> $\hat{\mathbf{x}}$, which is observed and passed, so $W(\hat{\mathbf{x}}_i)=1$ and it cancels from
> the per-object posterior. Detection does not cancel because it is not determined by data in hand.
> $P_{\rm pass}$ reappears in §4 and nowhere else.

---

## 3. The response identity, and Bartlett

For any statistic $f(\hat{\mathbf{x}})$ — the *function* fixed, only the density moving:

$$\partial_\gamma\mathbb E_\gamma[f]=\int f\,\partial_\gamma p
=\int f\,p\,\partial_\gamma\log p=\mathbb E_\gamma\big[f\,\hat s_\gamma\big]
\;\overset{\gamma=0}{=}\;\mathrm{Cov}_0\big(f,\hat s\big),\tag{M.4}$$

the last step using $\mathbb E_0[\hat s]=0$.

> **The average is over $\hat{\mathbf{x}}$**, under the marginal — not over $\theta$ under the
> posterior. Both $f$ and $\hat s$ are functions of the data.

Now take $f=\hat s$ itself. Its response *is* its variance:

$$\partial_\gamma\mathbb E_\gamma[\hat s]\big|_0=\mathrm{Var}_0(\hat s)\equiv\mathcal I
\qquad\Longrightarrow\qquad
\mathbb E_\gamma[\hat s]=\mathcal I\,\gamma+O(\gamma^2).\tag{M.5}$$

This is the whole trick: the score carries its own calibration. Every other statistic needs a
response supplied from outside; this one gets it from the scatter of the same numbers being summed.

**Complete sample.** If nothing is selected ($\Pi\equiv1$, so $A$ is already normalized), inverting
the straight line (M.5) gives, with $\sum_i\hat s_i$ estimating $N\mathcal I\gamma$ and
$\sum_i\hat s_i^2$ estimating $N\mathcal I$,

$$\hat\gamma=\frac{\sum_i\hat s_i}{\sum_i\hat s_i^{\,2}}.$$

The same $\mathcal I$ appears above and below and cancels, so $\partial_\gamma\mathbb E[\hat\gamma]=1$
by construction. No external responsivity, no $R_{\rm self}+R_{\rm blend}$ summed by hand.

---

## 4. Selection

A kept object is not drawn from $A$. It is drawn from $A$ restricted to the cut and **renormalized**:

$$p_{\rm keep}(\hat{\mathbf{x}}\mid\gamma)
=\frac{W(\hat{\mathbf{x}})\,A(\hat{\mathbf{x}}\mid\gamma)}{P(\text{keep}\mid\gamma)},
\qquad
P(\text{keep}\mid\gamma)=\int W A\,d\hat{\mathbf{x}}
=\mathbb E_{\pi(\theta)}\Big[\underbrace{P_{\rm pass}\,P_{\rm det}}_{\Pi}\big(S_\gamma\theta\big)\Big],$$

with $P_{\rm pass}(\theta)=\int_S L(\hat{\mathbf{x}}\mid\theta)\,d\hat{\mathbf{x}}$. One normalizer
handles detection *and* the cut together.

Take $\log$ and differentiate. $W$ is $\gamma$-free and equals $1$ for a galaxy in hand, so

$$\boxed{\;\hat s^{\rm keep}_i=\hat s_i-\langle s\rangle_{\rm sel}\;},
\qquad
\langle s\rangle_{\rm sel}\equiv\partial_\gamma\log P(\text{keep}\mid\gamma)\big|_0.$$

> **The normalizer subtracts; it does not divide inside the expectation.**
> $\partial_\gamma\log(A/P)=\partial_\gamma\log A-\partial_\gamma\log P$, and
> $P(\text{keep}\mid\gamma)$ is a *single number* — the double integral leaves no $\hat{\mathbf{x}}_i$
> and no $\theta$. So it comes out of $\mathbb E_{\rm post}$ entirely and is the **same constant
> subtracted from every galaxy in the catalogue**.

Evaluate it by pushing the same node bank through $S_\gamma$ and re-asking the classifier and
$P_{\rm pass}$ on the sheared nodes. Hold the Monte-Carlo draws inside $P_{\rm pass}$ fixed across
$\gamma$ and across nodes (common random numbers), or $P(\gamma)$ becomes a noisy step function.

**Apply (M.5) to the kept model.** $p_{\rm keep}$ is a properly normalized $\gamma$-family, so
Bartlett holds for *it*: $\mathbb E_{\rm keep}[\hat s^{\rm keep}]=0$ and
$\mathcal I_{\rm keep}=\mathrm{Var}(\hat s^{\rm keep})$. Inverting exactly as before:

$$\boxed{\;\hat\gamma=
\frac{\sum_i\big(\hat s_i-\langle s\rangle_{\rm sel}\big)}
{\sum_i\big(\hat s_i-\langle s\rangle_{\rm sel}\big)^{2}}\;}\tag{M.6}$$

> **Centre in both places.** Dropping $\langle s\rangle_{\rm sel}$ from the denominator while keeping
> it in the numerator — or dropping it from both — leaves an $O(1)$ bias, because a truncated sample
> has $\mathbb E_0[\hat s]\neq0$. §8 measures it: the uncorrected version is wrong by a factor of 19.

---

## 5. Derivation of `INFERENCE.md` (5.8)

Same model (M.1), same selection (§4). Only the denominator is obtained differently: from the
**curvature of the log-likelihood** rather than from $\mathrm{Var}(\hat s)=\mathcal I$.

**Definitions.** All derivatives at $\gamma=0$; $i$ indexes objects, $\theta$ prior nodes.

$$\varphi_\theta(\gamma)=\log\big[L(\hat{\mathbf{x}}_i\mid S_\gamma\theta)\,P_{\rm det}(S_\gamma\theta)\big],
\qquad
w(\theta)\propto L(\hat{\mathbf{x}}_i\mid\theta)\,P_{\rm det}(\theta)\,\pi(\theta),
\qquad
P(\gamma)=\mathbb E_\pi\big[\Pi(S_\gamma\theta)\big]$$

$$\hat s_i=\partial_\gamma\log A,\quad
\mathcal I_i=-\partial^2_\gamma\log A,\quad
\langle s\rangle_{\rm sel}=\partial_\gamma\log P,\quad
\mathcal I_{\rm sel}=-\partial^2_\gamma\log P$$

---

**Step 1 — the target is the MLE.** $\ell(\gamma)=\sum_i\log p_{\rm keep}(\hat{\mathbf{x}}_i\mid\gamma)$;
solve $\ell'(\hat\gamma)=0$. 

**Step 2 — one Newton step from $\gamma=0$.**

$$0=\ell'(\hat\gamma)\approx\ell'(0)+\hat\gamma\,\ell''(0)
\qquad\Longrightarrow\qquad
\hat\gamma=-\frac{\ell'(0)}{\ell''(0)}
=\frac{\sum_i\hat s_i^{\rm keep}}{\sum_i\mathcal I_i^{\rm keep}}\tag{M.7}$$

Slope over curvature. This is the only structural difference from §3, which instead linearized
$\mathbb E_\gamma[\hat s]=\mathcal I\gamma$ and inverted.

**Step 3 — selection splits both derivatives.** From §4,
$\log p_{\rm keep}=\log W+\log A-\log P(\gamma)$, with $W$ $\gamma$-free and $=1$ for a kept object:

$$\hat s_i^{\rm keep}=\hat s_i-\langle s\rangle_{\rm sel},\qquad
\mathcal I_i^{\rm keep}=\mathcal I_i-\mathcal I_{\rm sel}
\qquad\Longrightarrow\qquad
\hat\gamma=\frac{\sum_i\hat s_i-N\langle s\rangle_{\rm sel}}
{\sum_i\mathcal I_i-N\,\mathcal I_{\rm sel}}\tag{M.8}$$

(M.8) is `INFERENCE.md` (5.3). **The same subtraction is applied to both moments** — the step §3
never reaches, because that route forms no second derivative.

**Step 4 — $\hat s_i$, by Fisher's identity.** Differentiate (M.1), then divide by $A$; the ratio
$\pi LP_{\rm det}/A$ is the posterior $w$:

$$\partial_\gamma A=\int d\theta\;\pi\,LP_{\rm det}\;\partial_\gamma\log\big[LP_{\rm det}\big]
\qquad\Longrightarrow\qquad
\hat s_i=\mathbb E_w\big[\varphi'\big]\tag{M.9}$$

**Step 5 — $\mathcal I_i$, by Louis.** Apply
$\partial^2_\gamma\log A=\partial^2_\gamma A/A-(\partial_\gamma A/A)^2$ and push both derivatives
onto $LP_{\rm det}$, using $\partial^2_\gamma f=f[(\partial_\gamma\log f)^2+\partial^2_\gamma\log f]$:

$$\frac{\partial^2_\gamma A}{A}=\mathbb E_w\big[\varphi''+(\varphi')^2\big],
\qquad
\frac{\partial_\gamma A}{A}=\mathbb E_w\big[\varphi'\big]$$

$$\Longrightarrow\qquad
\partial^2_\gamma\log A=\mathbb E_w\big[\varphi''\big]+\mathrm{Var}_w\big(\varphi'\big)
\qquad\Longrightarrow\qquad
\mathcal I_i=-\mathbb E_w\big[\varphi''\big]-\mathrm{Var}_w\big(\varphi'\big)\tag{M.10}$$

The $\mathrm{Var}$ term is the leftover $-(\partial_\gamma A/A)^2$; equivalently, it is the
$\gamma$-dependence of the posterior weights themselves. Observed $=$ complete $-$ missing.

**Step 6 — the population pair.** $P(\gamma)=\int\Pi\,p_\gamma$ has the same form as $A$ with $\Pi$
in place of $LP_{\rm det}$, so Steps 4–5 apply verbatim:
$\langle s\rangle_{\rm sel}=(\log P)'(0)$, $\mathcal I_{\rm sel}=-(\log P)''(0)$.

**Step 7 — assemble.** (M.9), (M.10) and Step 6 into (M.8):

$$\boxed{\;\hat\gamma=
\frac{\displaystyle\sum_i\mathbb E_{w_i}\big[\varphi'\big]-N\,(\log P)'(0)}
{\displaystyle-\sum_i\Big(\mathbb E_{w_i}\big[\varphi''\big]+\mathrm{Var}_{w_i}\big(\varphi'\big)\Big)
+N\,(\log P)''(0)}\;}\tag{M.11}$$

which is `INFERENCE.md` (5.8). Every term is a first or second derivative of one of two scalar
curves: $\varphi_\theta$ per (object, node), and $\log P$ once for the population.

---

**Why it is unbiased at first order.** Bartlett on $p_{\rm keep}$ gives
$\mathbb E_0[\hat s^{\rm keep}]=0$, hence
$\mathbb E_\gamma[\hat s^{\rm keep}]=\mathcal I_{\rm keep}\gamma+O(\gamma^2)$ — numerator
$\approx N\mathcal I_{\rm keep}\gamma$. The information equality gives
$\mathbb E_0[\mathcal I_i^{\rm keep}]=\mathcal I_{\rm keep}$ — denominator
$\approx N\mathcal I_{\rm keep}$. The two cancel: $\partial_\gamma\mathbb E[\hat\gamma]=1+O(\gamma)$.

**Assumptions.**

| | |
|---|---|
| A1 | Differentiation under the integral sign, twice. |
| A2 | The Newton step is exact only to $O(\gamma^2)$. Iterate at $\hat\gamma$ to remove. |
| A3 | Correct specification. (M.11) is unbiased *for the model as written*; an external $R_b$ propagates linearly into $m$. |
| A4 | $W$ is deterministic in $\hat{\mathbf{x}}$, so it cancels per object; $P_{\rm det}$ is not, so it does not (§2). |
| A5 | $\pi$ available as a sampler; node-bank ESS adequate for both $\mathbb E_w$ and $\mathbb E_\pi$. |
| A6 | Common random numbers inside $P_{\rm pass}$, across $\gamma$ *and* across nodes — otherwise $(\log P)''$ is noise. |

**Contrast with (M.6), in one line.** Both are (M.7). (M.6) replaces
$\sum_i\mathcal I_i^{\rm keep}$ by $\sum_i(\hat s_i^{\rm keep})^2$, legitimate because
$\mathbb E_0[(\hat s^{\rm keep})^2]=\mathbb E_0[\mathcal I^{\rm keep}]$ — the information equality.
§7 quantifies what that substitution costs.

---

## 6. The Eulerian form: derivation of `INFERENCE.md` (5.3)

Shear acts on the **prior**: the samples stay put and the density moves. The denominator comes from
the **curvature** of the log-likelihood, by Louis's identity. Self-contained: everything the
derivation needs, including the selection normalizer of §4, is restated below.

**Definitions.** All derivatives at $\gamma=0$; $i$ indexes objects, $\theta=(\mathbf{x},\mathbf{n})$
prior nodes, $p_\gamma=S_\gamma\#p_0$ the pushforward of the prior along the shear map.

$$u_\gamma\equiv\partial_\gamma\log p_\gamma,\quad u\equiv u_0,
\qquad
w(\theta)\propto L(\hat{\mathbf{x}}_i\mid\theta)\,P_{\rm det}(\theta)\,p_0(\theta),
\qquad
\Pi=P_{\rm pass}P_{\rm det},
\qquad
P(\gamma)=\mathbb E_{p_\gamma}[\Pi]$$

$$\hat s_i=\partial_\gamma\log A,\quad
\mathcal I_i=-\partial^2_\gamma\log A,\quad
\langle s\rangle_{\rm sel}=\partial_\gamma\log P,\quad
\mathcal I_{\rm sel}=-\partial^2_\gamma\log P$$

---

**Step 1 — the model.** The flow and the classifier, integrated against the sheared prior:

$$A(\hat{\mathbf{x}}\mid\gamma)=\int d\theta\;L(\hat{\mathbf{x}}\mid\theta)\,P_{\rm det}(\theta)\,
p_\gamma(\theta),\tag{M.12}$$

`INFERENCE.md` (1.1) with detection kept. $L$ and $P_{\rm det}$ are fixed functions of truth, so
$p_\gamma$ is the only factor of the integrand that is ever differentiated. $A$ is **not
normalized** — $\int A\,d\hat{\mathbf{x}}=P(\text{det}\mid\gamma)$, which itself moves with $\gamma$;
Step 5 supplies the normalizer.

**Step 2 — the generator.** A pushforward obeys continuity,
$\partial_\gamma p_\gamma+\nabla\!\cdot\!(p_\gamma v)=0$, with $v\equiv\partial_\gamma S_\gamma|_0$
the shear velocity field on truth. Divide by $p_\gamma$ and expand the divergence:

$$\boxed{\;u=-\Big(v\cdot\nabla\log p_0+\nabla\!\cdot\!v\Big)\;}\tag{M.13}$$

`INFERENCE.md` (2.7), and the price of putting $\gamma$ on the prior: **$\nabla\log p_0$ over the
whole scene** — shape, size, flux, Sérsic and neighbour configuration together — plus the
$\nabla\!\cdot\!v$ contributed by the change of measure.

> **Composition is not pushforward.** $p_\gamma=p_0\circ S_{-\gamma}$ with no Jacobian gives
> $u=-v\cdot\nabla\log p_0$ and drops $-\nabla\!\cdot\!v$ — not a small term: for the shape channel
> in the reduced-shear convention $v=(1-\varepsilon^2,\,i(1+\varepsilon^2))$ gives
> $\nabla\!\cdot\!v=-4\varepsilon_1$ for the $g_1$ flow, the same order as what is kept.

**Step 3 — $\hat s_i$, by Fisher's identity.** Differentiate (M.12), divide by $A$; the ratio
$LP_{\rm det}p_0/A$ is the posterior $w$:

$$\partial_\gamma A=\int d\theta\;LP_{\rm det}\;p_\gamma\,\partial_\gamma\log p_\gamma
\qquad\Longrightarrow\qquad
\hat s_i=\mathbb E_{w_i}\big[u\big]\tag{M.14}$$

`INFERENCE.md` (2.2). **$u$ does not depend on the galaxy** — the structural payoff. The derivative
falls on the prior alone, so $u_k$ and $\partial_\gamma u_k$ are built **once per node** and shared
across the catalogue, leaving one flow call $L_k=L(\hat{\mathbf{x}}_i\mid\theta_k)$ as the only
per-(object, node) quantity.

**Step 4 — $\mathcal I_i$, by Louis.** Apply
$\partial^2_\gamma\log A=\partial^2_\gamma A/A-(\partial_\gamma A/A)^2$ and push both derivatives
onto the prior, using $\partial^2_\gamma p_\gamma=p_\gamma\big[u_\gamma^2+\partial_\gamma u_\gamma\big]$:

$$\frac{\partial^2_\gamma A}{A}=\mathbb E_w\big[\partial_\gamma u+u^2\big],
\qquad
\frac{\partial_\gamma A}{A}=\mathbb E_w[u]$$

$$\Longrightarrow\qquad
\partial^2_\gamma\log A=\mathbb E_w\big[\partial_\gamma u\big]+\mathrm{Var}_w(u)
\qquad\Longrightarrow\qquad
\mathcal I_i=-\mathbb E_{w_i}\big[\partial_\gamma u\big]-\mathrm{Var}_{w_i}(u)\tag{M.15}$$

The $\mathrm{Var}$ term is the leftover $-(\partial_\gamma A/A)^2$ — observed $=$ complete $-$
missing. $\partial_\gamma u|_0=\partial^2_\gamma\log p_\gamma|_0$ is a second derivative of the
**prior**; the flow is never differentiated with respect to $\gamma$, which is not one of its inputs.

**Step 5 — selection, and the estimator.** A kept object is drawn from $A$ restricted to the cut and
renormalized (§4), so

$$\log p_{\rm keep}=\log W+\log A-\log P(\gamma),
\qquad
P(\gamma)=\mathbb E_{p_\gamma}\big[\Pi\big],\quad\Pi=P_{\rm pass}P_{\rm det}.$$

$W$ is $\gamma$-free and equals $1$ for a galaxy in hand, so differentiating once and twice centres
**both** moments by the same population quantities:

$$\hat s^{\rm keep}_i=\hat s_i-\langle s\rangle_{\rm sel},
\qquad
\mathcal I^{\rm keep}_i=\mathcal I_i-\mathcal I_{\rm sel}.$$

The target is the MLE, $\ell'(\hat\gamma)=0$ with $\ell=\sum_i\log p_{\rm keep}$. One Newton step
from $\gamma=0$ — slope over curvature — gives

$$\hat\gamma=-\frac{\ell'(0)}{\ell''(0)}
=\frac{\sum_i\hat s^{\rm keep}_i}{\sum_i\mathcal I^{\rm keep}_i}
=\frac{\sum_i\hat s_i-N\langle s\rangle_{\rm sel}}{\sum_i\mathcal I_i-N\,\mathcal I_{\rm sel}}$$

Centring the numerator alone, or neither, leaves an $O(1)$ bias: a truncated sample has
$\mathbb E_0[\hat s]\neq0$.

**Step 6 — the population pair.** $P(\gamma)=\int\Pi\,p_\gamma$ has exactly the shape of (M.12) with
$\Pi$ in place of $LP_{\rm det}$, so Steps 3–4 apply verbatim with the posterior replaced by the
$\Pi$-weighted prior:

$$\langle s\rangle_{\rm sel}=\mathbb E_\Pi\big[u\big],
\qquad
\mathcal I_{\rm sel}=-\mathbb E_\Pi\big[\partial_\gamma u\big]-\mathrm{Var}_\Pi(u)\tag{M.16}$$

the second being `INFERENCE.md` (5.3b). Nothing new has to be evaluated: $u_k$, $\partial_\gamma u_k$
and $\Pi_k$ are already in the node bank, so (M.16) is one extra weighted sum over nodes —
$O(N_{\rm node})$ against the estimator's $O(N_{\rm gal}N_{\rm node})$, i.e. free.

**Step 7 — assemble.** (M.14), (M.15) and (M.16) into Step 5:

$$\boxed{\;\hat\gamma=
\frac{\displaystyle\sum_i\mathbb E_{w_i}[u]\;-\;N\,\mathbb E_\Pi[u]}
{\displaystyle-\sum_i\Big(\mathbb E_{w_i}\big[\partial_\gamma u\big]+\mathrm{Var}_{w_i}(u)\Big)
\;+\;N\Big(\mathbb E_\Pi\big[\partial_\gamma u\big]+\mathrm{Var}_\Pi(u)\Big)}\;}\tag{M.17}$$

which is `INFERENCE.md` (5.3). Numerator and denominator are the **same pair of functionals** — the
mean of $u$, and Louis's combination — evaluated first at the posterior weights $w_i$, then again at
the prior-survival weights $\Pi$, and subtracted.

> **Where $R_{\rm blend}$ enters.** With $\mathbf{n}$ latent, $u$ carries neighbour components —
> separation, neighbour shape and size, all spin-2 — so $\hat s_i=\sum_kw_ku_k$ picks the blend
> response up with no separate term. It vanishes exactly when $L_k$ is blind to the neighbour
> position angle: then that angle's posterior equals its prior and isotropy kills the sum (§3).

**Why it is unbiased at first order.** $p_{\rm keep}$ is a properly normalized $\gamma$-family, so
Bartlett applies to it: $\mathbb E_0[\hat s^{\rm keep}]=0$, hence
$\mathbb E_\gamma[\hat s^{\rm keep}]=\mathcal I_{\rm keep}\gamma+O(\gamma^2)$ and the numerator is
$\approx N\mathcal I_{\rm keep}\gamma$. The information equality gives
$\mathbb E_0[\mathcal I^{\rm keep}_i]=\mathcal I_{\rm keep}$, so the denominator is
$\approx N\mathcal I_{\rm keep}$. The two cancel:
$\partial_\gamma\mathbb E[\hat\gamma]=1+O(\gamma)$.

**Step 8 — the same $A$ from shearing the samples.** A pushforward is defined so that
$\int f(\theta)\,p_\gamma(\theta)\,d\theta=\mathbb E_{\theta'\sim p_0}\big[f(S_\gamma\theta')\big]$
for any $f$; take $f=LP_{\rm det}$:

$$\int d\theta\;L(\hat{\mathbf{x}}\mid\theta)P_{\rm det}(\theta)\,p_\gamma(\theta)
=\int d\theta'\;p_0(\theta')\,L\big(\hat{\mathbf{x}}\mid S_\gamma\theta'\big)\,
P_{\rm det}\big(S_\gamma\theta'\big)\tag{M.18}$$

So the derivation can be read either way round — $\gamma$ on the prior, or $\gamma$ pushed through
the samples with $p_0$ held fixed. $A$ is *literally the same function of $\gamma$*, so
$\hat s_i,\mathcal I_i,\langle s\rangle_{\rm sel},\mathcal I_{\rm sel}$ are the same four numbers
either way. What the choice buys is that here the derivative lands on the prior alone, so $u_k$ and
$\partial_\gamma u_k$ amortize over the catalogue; what it costs is $\nabla\log p_0$ in (M.13).

---

**Under isotropy the numerator's correction vanishes and the denominator's does not.** Shear is
spin-2: conjugating by a rotation of angle $\alpha$ carries $S_\gamma$ to $S_{\gamma e^{2i\alpha}}$.
If $p_0$ is isotropic and $\Pi$ rotationally invariant, $P(\gamma)$ is constant along that circle, so
$P(\gamma)=P(|\gamma|)$ — and a smooth function of $|\gamma|$ has zero gradient at the origin:

$$\langle s\rangle_{\rm sel}=0\ \ \text{exactly},
\qquad
\mathcal I_{\rm sel}=\iota\,\mathbb 1_{2\times2}\neq0,$$

the second because the only rotation-invariant symmetric $2\times2$ tensor is a multiple of the
identity. The whole selection correction then sits in the **denominator**, as one number $\iota$: of
the two, the familiar first-order centring is the one that does nothing. Keep it in (M.17) anyway —
PSF anisotropy, any cut touching $\hat e$, and position-dependent depth or masking break the
symmetry.

> **No conflict with §4 or §8(b),** where dropping $\langle s\rangle_{\rm sel}$ was catastrophic:
> those use a 1D *shift* toy, whose map is spin-0. The vanishing above is a property of spin-2, not
> of selection.

**Regularity.** Differentiation under the integral sign, twice; the Newton step of Step 5 is exact
only to $O(\gamma^2)$, removable by iterating at $\hat\gamma$; and the Monte-Carlo draws inside
$P_{\rm pass}$ must be held fixed across $\gamma$ *and* across nodes, or $\partial^2_\gamma\log P$ is
noise.

**Requirements** — `INFERENCE.md` §5B.3.

| | |
|---|---|
| E1 | $\nabla\log p_0$ over the whole scene — shape, size, flux, Sérsic, neighbour configuration — differentiable. An analytic isotropic shape prior covers the shape channel only. |
| E2 | A **scene** prior, hence clustering statistics. $\mathbf{n}$ marginalized, never substituted from measurements (errors-in-variables); correct, but it discards the neighbours one can see. |
| E3 | $L$ conditioned on PSF size, PSF ellipticity and depth, or it is wrong across a varying footprint. |
| E4 | $P_{\rm det}(\theta)$ **and** its integral against the sheared prior — over objects never observed. |
| E5 | ESS per object, not node count. $\hat s_i=\mathbb E_{w_i}[u]$ is a ratio of weighted sums, hence **biased, not merely noisy, at small ESS**, with the bias common across objects — catalogue averaging does not remove it. |

---

## 7. Relation to `INFERENCE.md` (5.3)

Same model, same numerator, different denominator.

| | this document, (M.6) | `INFERENCE.md` §5B, (5.3) $=$ (M.17) |
|---|---|---|
| numerator | $\sum_i\hat s_i-N\langle s\rangle_{\rm sel}$ | same |
| denominator | $\sum_i(\hat s_i-\langle s\rangle_{\rm sel})^2$ | $\sum_i\mathcal I_i-N\mathcal I_{\rm sel}$ |
| information from | Bartlett: variance of the score | Louis: $-\mathbb E_{w_i}[\partial_\gamma u]-\mathrm{Var}_{w_i}(u)$ |
| prior derivatives | first only, $u$ (M.13) | first **and** second, $u$ and $\partial_\gamma u$ |
| extra cost | — | one more derivative **per node**, amortized $\Rightarrow$ free |

The prior-derivative row compares the two on the *Eulerian* footing. Computed the other way (§5),
(M.6) needs no prior derivative at all — but that is §5's advantage, not Bartlett's.

**Against (5.3), cost is not the reason to prefer (M.6).** On the Lagrangian route $\varphi''$ is
per (object, node), so second order costs $\approx$ 2–3$\times$; here $\partial_\gamma u_k$ is per
**node**, so (M.17)'s denominator adds $O(N_{\rm node})$ to an $O(N_{\rm gal}N_{\rm node})$
estimator. What (M.6) buys against (5.3) is a **free internal check** — `INFERENCE.md` §5B.2's
consistency test *is* the statement that the two denominators agree in expectation,

$$\frac1N\sum_i\big(\hat s_i-\langle s\rangle_{\rm sel}\big)^2
\;\approx\;\frac1N\sum_i\mathcal I_i-\mathcal I_{\rm sel},$$

both sides estimating the kept population's information. Run with the uncorrected denominator and
they disagree by exactly the missing $\mathcal I_{\rm sel}$. Under the isotropy of §6 the centring
drops out of the left side and the test is $N^{-1}\sum_i\hat s_i^{\,2}$ against
$N^{-1}\sum_i\mathcal I_i-\iota$.

**They agree at $\gamma=0$ — that agreement *is* the information equality — but drift apart away
from it,** and the drift is computable from (M.4). Both denominators are averages of fixed functions
of the data, so apply (M.4) to each with the kept-sample score $\hat s^{\rm keep}$:

$$\partial_\gamma\mathbb E_\gamma\big[(\hat s^{\rm keep})^2\big]_0=\mu_3,
\qquad
\partial_\gamma\mathbb E_\gamma\big[\mathcal I^{\rm keep}\big]_0
=\mathrm{Cov}_0\big(\mathcal I^{\rm keep},\hat s^{\rm keep}\big),$$

$\mu_3$ being the third central moment of the kept-sample score. Dividing the two,

$$\boxed{\;m_{\rm(M.6)}-m_{\rm(5.3)}\;=\;
-\gamma\,\frac{\mu_3-\mathrm{Cov}_0\big(\mathcal I^{\rm keep},\,\hat s^{\rm keep}\big)}
{\mathcal I_{\rm keep}}\;+\;O(\gamma^2)\;}\tag{M.19}$$

(equivalently $m_{(5.8)}$: by (M.18) the two share this denominator, and §8 labels it (5.8)).

**The gap is set by how much $\mathcal I_i$ varies across objects.** If every object carried the same
information the covariance would vanish and (M.19) would reduce to
$-\gamma\,\mathrm{skew}(\hat s)\sqrt{\mathcal I_{\rm keep}}$ — a Gaussian special case, not the
general result. Real flows give a wide spread of $\mathcal I_i$; §8(c) shows the covariance term
changing the prediction by 60%.

**But it is only a step size.** Both are a *single Newton step* from $\gamma=0$ on the same
log-likelihood, differing in which curvature estimate sets the step length, and the root is fixed by
the numerator they share. Iterating either — re-evaluating $\langle s\rangle_{\rm sel}$ and the
denominator at $\hat\gamma$ rather than at $0$ — drives both to the **same** MLE and (M.19)
disappears. Use (M.6) once as a cheap cross-check on (5.3); iterate it to use it on its own.

---

## 8. Numerical checks

**(a) The $P_{\rm det}$ channel of (M.3) is not optional.** Toy: $\theta\sim\mathcal N(0,1)$,
$S_\gamma\theta=\theta+\gamma$, Gaussian flow $\sigma=0.7$, sigmoid $P_{\rm det}$. Deterministic
quadrature, so the only error is finite differencing ($\sim10^{-6}$):

| $\hat{\mathbf{x}}$ | direct $\partial_\gamma\log A$ | Eulerian | (M.3) **without** $P_{\rm det}$ | (M.3) as written |
|---|---|---|---|---|
| $-0.5$ | $-0.095938$ | $-0.095938$ | $-0.824617$ | $-0.095938$ |
| $+0.4$ | $+0.445618$ | $+0.445618$ | $\mathbf{-0.093097}$ | $+0.445618$ |
| $+1.2$ | $+0.930325$ | $+0.930325$ | $+0.550358$ | $+0.930325$ |
| $+2.0$ | $+1.423899$ | $+1.423899$ | $+1.175716$ | $+1.423899$ |

The omitted channel is 17–760% of the score and reverses its sign at $\hat{\mathbf{x}}=+0.4$.

**(b) (M.6) under a cut.** `INFERENCE.md` A.7 toy: $\tau=1.0$, $\sigma=0.7$, cut $\hat y>0.3$, truth
$\gamma=0.05$; 40 seeds $\times$ $2\times10^6$ draws.

| estimator | $\hat\gamma$ | multiplicative bias $m$ |
|---|---|---|
| uncorrected, $\sum\hat s/\sum\hat s^2$ | $+0.93673\pm0.00011$ | $+1770\%$ |
| **(M.6)** | $+0.04837\pm0.00038$ | $-3.3\%\pm0.8\%$ |
| (5.8), Louis | $+0.04958\pm0.00040$ | $-0.8\%\pm0.8\%$ |

Errors are on the mean of 40 seeds. Both corrected estimators work; centring is what matters. The
individual $m$ values here are too noisy to separate the two — for that, pair them.

**(c) (M.19), the difference between the two denominators.** Both estimators share a numerator, so
evaluating them on the *same* catalogue cancels the common noise and measures their ratio precisely.

*Gaussian toy, where $\mathcal I_i$ is the same for every object* ($\mathrm{Cov}=0$ identically):
$\mu_3=0.1046$, $\mathcal I_{\rm keep}=0.2102$, $\mathrm{skew}=1.083$, so (M.19) predicts
$-\gamma\times0.4965$. 24 seeds $\times$ $4\times10^6$ draws:

| $\gamma$ | measured | (M.19) |
|---|---|---|
| 0.02 | $-1.028\%\pm0.029\%$ | $-0.993\%$ |
| 0.05 | $-2.534\%\pm0.028\%$ | $-2.482\%$ |
| 0.10 | $-5.074\%\pm0.027\%$ | $-4.965\%$ |
| 0.20 | $-10.228\%\pm0.025\%$ | $-9.930\%$ |

*Mixture prior with a dilation shear map, where $\mathcal I_i$ genuinely varies*
($\mathrm{sd}/\mathrm{mean}=1.14$): $\mu_3=5.355$, $\mathrm{Cov}=1.944$,
$\mathcal I_{\rm keep}=1.448$. 12 seeds $\times$ $2\times10^6$ draws:

| $\gamma$ | measured | (M.19) full | (M.19) **without** the $\mathrm{Cov}$ term |
|---|---|---|---|
| 0.02 | $-4.653\%\pm0.058\%$ | $-4.713\%$ | $-7.398\%$ |
| 0.05 | $-11.446\%\pm0.054\%$ | $-11.782\%$ | $-18.496\%$ |
| 0.10 | $-22.124\%\pm0.047\%$ | $-23.565\%$ | $-36.992\%$ |

Linear in $\gamma$ in both models and matching (M.19) to a few percent, the residual being the dropped
$O(\gamma^2)$. Dropping the covariance term overpredicts by 60% — it is not a refinement. Both runs
also confirm the information equality itself: $0.210203$ vs $0.210238$, and $1.447722$ vs
$1.447593$.

---

## 9. Pointers

`INFERENCE.md` §2.3 (Bartlett and Louis), §2.4 (the generator (2.7), and the channel velocities
$v$ that §6 needs), §5B.1 (the Eulerian algorithm and the detection-vs-cut asymmetry),
§5B.2 ($\mathcal I_{\rm sel}$, the isotropy argument, and the consistency test), §5B.3 (what the
Eulerian route demands, $\nabla\log p_0$ first among them), §5C (the Lagrangian parametrization and
its algorithm), (5.9) (where (M.6) is recorded there).

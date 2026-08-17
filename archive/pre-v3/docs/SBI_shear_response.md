# Addendum: Response-aware Training Update

> **Current framing lives in [`GOALS.md`](GOALS.md)** — the response loss now supervises
> three derivatives (shape, measured mag/size → selection, detection), all on primary-only
> matched pairs. The mechanism below (shear via the analytic distortion, response in the
> loss) is unchanged; `GOALS.md` wins on targets, data, and cuts.

This section supersedes parts of the original training discussion while keeping the
Bayesian forward-model framework unchanged.

## Philosophy

The probabilistic model remains

$$
p_\theta(\hat{\mathbf{x}}\mid\mathbf{x}),
$$

where $\mathbf{x}$ denotes the latent true scene and $\hat{\mathbf{x}}$ the measured
catalogue quantities. Shear still enters only through the analytic distortion
$S_\gamma(\mathbf{x})$, exactly as in the original document.

The key refinement is that **the scientific objective is not merely to maximize the
likelihood accuracy**. Weak-lensing cosmology is primarily sensitive to the
**first-order shear response**. Therefore the training objective should explicitly
preserve the Jacobian of the learned forward model with respect to shear.

## Response-aware (Sobolev) training

For each underlying galaxy configuration we generate paired realizations at
$\gamma=\pm\delta$ (or several nearby shears).

These paired simulations provide

- function values:
  $$
  (\gamma,\hat{\mathbf{x}})
  $$
- finite-difference estimates of the local response:
  $$
  \frac{\Delta\hat{\mathbf{x}}}{\Delta\gamma}.
  $$

The network predicts the forward likelihood as before, while autograd computes its
local derivative through the analytic shear map.

The objective becomes

$$
\mathcal L
=
\mathcal L_{\rm NLL}
+
\lambda\,
\mathcal L_{\rm response},
$$

where

$$
\mathcal L_{\rm response}
=
\left\|
\frac{\partial p_\theta}{\partial\gamma}
-
\frac{\partial p_{\rm sim}}{\partial\gamma}
\right\|^2.
$$

Equivalently, one may formulate the response loss using induced catalogue
statistics (e.g. mean measured ellipticity or selection probability) rather than
the density itself.

The important point is that the paired simulations supervise the **local Jacobian**
rather than only the likelihood values.

## Interpretation

The probabilistic model is unchanged.

Originally:

- Learn the likelihood accurately.
- Hope the induced shear response is accurate.

Revised:

- Learn the same likelihood.
- Explicitly constrain its first-order shear response during training.

This changes the optimization objective, not the Bayesian formulation.

## Role of multi-shear simulations

Multi-shear realizations are no longer viewed primarily as additional likelihood
training data.

Instead they serve three purposes:

1. supervise the local shear Jacobian;
2. validate autograd against finite differences;
3. verify end-to-end shear recovery.

The $g=0$ realization still contains essentially all information required to
learn the forward density, while nearby shears primarily constrain the response.

## Scientific motivation

A model with slightly worse global negative log-likelihood but significantly
better shear response is scientifically preferable.

Therefore likelihood quality alone should not be the primary benchmark.
The principal benchmark is recovery of the correct first-order response,
equivalently unbiased multiplicative shear calibration.

---

## Parked idea: per-object SNC pair-response regression (not yet implemented, 2026-07-05)

**Context.** The current coherent calibration ("Path B", `crowdcoh`) supervises the
flow's shear response with a *binned* target: the measured coherent response
`R_sim` is averaged inside a 6×3×5 grid of (flux × size × crowd=`r_blend`) cells,
and the response loss pulls `R_model(cell) → R_sim(cell)`. Held-out gold-constant
gives global **m = −0.73% ± 0.41%** — sub-percent — but the per-`R_blend`-bin
residual is still **±6%** (ISO −2.7% / q1 −3.7% / q2 −0.0% / q3 −5.8% / q4 +6.4%),
so the global is partly a *cancellation* across bins, not a genuinely flat
per-property calibration. The bin residual is the coarse grid failing to track
within-cell response variation.

**The idea.** Replace the binned target with a **per-object** response-regression
target, using the two-sided antithetic SNC pairs we already render:

- For each object detected+converged at **both** +g and −g of the constant render
  (mutual-detection subset), the per-object coherent pair-response is
  `r_sim_i = (e_+ − e_−)·ĝ / (2g)`. Shared noise ⇒ intrinsic shape *and* noise
  cancel, so `r_sim_i` is a low-variance per-object estimate.
  (This is exactly the per-object `proj/g` already computed inside
  `compute_response_target_constant.py` *before* it bins.)
- Train the flow's response loss as a continuous per-row regression
  `λ·(R_model_i − r_sim_i)²` instead of the per-bin mean-matching. The flow's
  response is already a smooth function of the conditioning features
  (flux, size, `nbr_flux`/`r_blend`), so it interpolates — no bin edges, no
  starved cells, crowd features used continuously.

**Why it should help.** Removes the bin-edge discretisation that produces the ±6%
per-bin tilt → aims for genuinely flat per-property sub-percent m, robust to a
shift in the crowding distribution (survey depth) that the current cancellation is
*not* robust to. Also naturally separates m from c: `(e_+ − e_−)/2g → m`,
`(e_+ + e_−)/2 → c` (additive/PSF-leakage diagnostic).

**Legitimacy.** Still the sanctioned response-aware calibration — a
property→response map fit on sheared sims and **cross-validated across cases**
(target on cases 0–19, validate held-out 20–39), *not* dividing measured m out of
the validation set.

**Caveat — the selection term.** Restricting to mutual-detection yields the *pure
measurement* response (selection made shear-independent). The full selected
sample's response also carries a **selection-response** piece (objects
entering/leaving the sample with shear); that must be supplied separately by the
detection/selection classifier. Forgetting it re-introduces bias. This is a clean
measurement-vs-selection decomposition (measurement→flow, selection→classifier),
not a free lunch.

**Implementation sketch (when picked up).** (1) Save the per-object `r_sim_i`
(+ `case`, `input_index`, features) from `compute_response_target_constant.py`
rather than the binned grid. (2) In `train_measurement_model.py`, carry a per-row
target tensor through `_resp_loader` (alongside `ww`) and switch `epoch_response`
from per-bin mean-matching to a per-row weighted MSE. The response loss can be
evaluated on the constant-render galaxies directly (their own features + `r_sim_i`),
while the NLL stays on the g=0 catalogue — a two-dataset loop, avoiding any need to
match g=0 rows to constant rows by `input_index`.

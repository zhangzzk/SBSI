# Gold-V3 — Folding R_blend into the flow

**Status:** DESIGN, updated 2026-07-28 after owner discussion. **Nothing built.** Supersedes the
2026-07-28 IDEA draft (kept below where still correct, marked where retracted). Builds on
`Gold-V2.md` (R_blend deliberately external) and `INFERENCE.md` (§3, §5B, §5C.3).

## The idea

Make the flow generate its own blend response, so the external BlendEMU term disappears and the
likelihood is complete.

Immediate motivation from `WORKLOG.md` 2026-07-28i..l: BlendEMU under-predicts the per-pair blending
response by **−41.5% below 1″** inside the deliverable domain (5.7σ, held-out −44.6%, null test
0.6σ). Neither in-domain retraining (−40.3%) nor close-pair loss weighting (−37.3% at 4×, −37.1% at
13×, saturating while degrading 1–2″) moves it — a representational limit, not an incentive one. So
the external term is the weakest component in `m = R_sim/(R_flow + R_blend) − 1`, and R_flow is
clean (self-response error −0.05% over 5.9M galaxies).

## Decision: TWO flows, not one — for now

| | role | status |
|---|---|---|
| **flow #1** | the certified self-response flow (`meas_szfl_noz_lam450_fixresp`) | **UNTOUCHED** |
| **flow #2** | new blend flow; a firewalled drop-in replacement for BlendEMU | to build |

Rationale:

- **Firewall.** Gold-v1 sits at m=+0.245%. Every attempt to add capacity to it has moved it the
  wrong way (the DeepSets trunk cost −5.24%, cont.146). Flow #2 is a component swap inside the
  *existing* additive frame; if it fails, nothing regressed.
- **Decoupled loss weights.** The self-response label is clean; the blend label is ~20× noisier per
  pair (below). Balancing `λ_self` against `λ_blend` in one loss is exactly the tuning that produced
  nothing on 2026-07-28d..g. Two flows means neither term can starve the other.
- **Different rows.** The self target lives on the primary-only-sheared leg, the blend target on the
  both-sheared leg. One flow would need two disjoint sub-batches with different supervision.

**Two flows is the de-risked step, not the endpoint.** Under Direction B, `I = Var(s)` needs
`Cov(s_self, s_nbr)`, and two separate densities cannot produce a cross term. Merge when Direction B
goes live.

## Flow #2 specification

```
p(measured e | primary true props, ONE true neighbour, separation VECTOR, flux shells near/mid/far)
```

- **Primary true shape is KEPT** (decided 2026-07-28). It is the dominant predictor of the measured
  shape — remove it and the mean head drifts to the population average and the density is worthless
  for the eventual merge and for `Var(s)`. It also carries the primary-orientation-versus-pair-axis
  channel, which is half the spin-2 geometry. No leakage risk: the primary's shape is uncorrelated
  with the neighbour's shear direction, so it cannot absorb or fake the blend signal.
- **Neighbour:** true (oriented) shape, size, flux.
- **Separation: vector or scalar is OPEN** — it depends entirely on the position-shear question
  below. See "Do we need the pair angle?".
- **Flux shells near/mid/far** aggregate the rest of the crowding. This is what lets a pair-trained
  model be applied to a real scene at all. Note `nbr_flux_mid` **does not exist yet**; the code has
  `nbr_flux_near`, `nbr_flux_far`, `nbr_flux_max` (`measurement_model._MEAS_CROWD_NBR`).
- **Used only through its neighbours-only derivative.** Shear the neighbour, hold the primary fixed,
  read the mean-head shift. Summed over neighbours at evaluation.

For contrast, the certified flow #1's entire neighbour conditioning is those three scalar fluxes —
no neighbour size, no separation, no angle, no neighbour shape. Flow #2's conditioning is a genuine
addition, not a tweak.

## The blocker: conditioning must not be spin-0 — RESOLVED by the spec above

All of flow #1's conditioning is scalar, so at γ=0 the likelihood is invariant under rotating a
neighbour about the primary ⇒ `Cov(ê, s_nbr) ≡ 0` identically (INFERENCE.md §3). **No loss term can
fix this** — a response penalty cannot force a response through a channel the architecture
symmetrises away.

Concretely: size, flux and scalar separation are all shear-invariant. If those were flow #2's only
neighbour features, the shifted and unshifted inputs would be *identical*, the induced response
exactly zero for any parameters, and the new loss term a constant with zero gradient. It could not
teach anything.

Resolved by including the neighbour's **true shape**, which shears directly via the Möbius map
already in `shifted_feature_frame`. That channel alone is enough to make the response non-zero, so
the spin-2 blocker does *not* by itself force the pair angle into the conditioning.

### Do we need the pair angle?

**Two separate jobs, and they have different answers.**

*As an accuracy variable:* no. If neighbours are isotropically distributed about the primary, then
`E[R_blend | scalars]` is exactly the angle-average, which is what enters `m`. Dropping φ costs
scatter, not bias — the same argument that retracts the missing-orientation story below. Scalar
separation suffices.

*As a response channel:* **only if the sim shears positions.** If it does, φ is structurally
required, for two reasons that isotropy does not rescue:

1. Blend contamination pulls the measured shape **along the pair axis**: `e_bias ≈ A(d)·(cos2φ,
   sin2φ)`. A model blind to φ cannot point `∂μ/∂d` in that direction at all, so the geometric part
   of the response has nowhere to live and gets faked through the shape channel.
2. The position contribution **does not average away**. Under a position shear `δd/d = γ·cos2(φ−φ_γ)`
   and `δφ ∝ −γ·sin2(φ−φ_γ)`. Projecting the induced shape change onto ĝ_s gives

   ```
   stretch term:  A'(d)·d·γ·cos²2(φ−φ_γ)      <cos²> = 1/2
   rotation term: 2·A(d)·γ·sin²2(φ−φ_γ)       <sin²> = 1/2
   ```

   Both enter as **cos²/sin², not cos/sin** — the φ-dependence appears twice and the product
   survives isotropic averaging at half strength. This is the trap: "isotropic ⇒ average it away"
   is right for a term linear in cos2Δ and wrong for these.

Note the distinction between *constructing the shift* and *conditioning on φ*. The per-row φ is known
in the catalogue, so the correct per-row `δd` can always be applied even if φ is not a feature — but
without φ as a feature the model still cannot orient the resulting shape change. So the feature is
the binding requirement, not the shift.

**Therefore:** if job 15328296 finds shape-only shear, the neighbour's oriented shape is the whole
channel, scalar separation is enough, and the pair angle can be dropped. If it finds position shear,
the pair angle goes in. This is the single question that settles the conditioning set.

(Separate from `r_blend`-as-a-feature, which modulates the primary's *own* response with crowding.
That lives in `R_self` and works as intended.)

## RETRACTED — the double-counting argument

An earlier version of this discussion claimed flow #2's response "is `R_self + R_blend`, so you
cannot add it to flow #1". **That was wrong.** The response is whatever you choose to differentiate,
and flow #2's training catalogue has the primary unsheared, so the only derivative anyone would take
is the neighbours-only one. There is no double-count to avoid.

What survives is much smaller: flow #2 retains a nonzero, *unsupervised* derivative with respect to
the primary's true shape, simply because that shape is in the conditioning. It is a direction that
must never be read — a comment in the code, not a design problem.

## Loss structure — a mirror, but NOT a symmetric one

|  | flow #1 (primary-only sheared) | flow #2 (neighbour-only sheared) |
|---|---|---|
| self-response term | target = measured `R_self` | **open — see below** |
| blend-response term | target = 0 | target = measured `R_blend` |
| θ coupling (mag/size, dims 2,3) | pinned (`theta_coupling_residual`) | mirrored |
| other properties vs neighbour shear | — | pinned 0 (minor) |

### OPEN QUESTION: should flow #2's self-response be pinned to zero?

Pinning it is *not* the mirror image of pinning flow #1's blend response to zero, because the two
lies are not the same size:

- flow #1, blend → 0: the true blend response is ≈ **0.012**. A small misstatement.
- flow #2, self → 0: the true self response is ≈ **0.3**, and it *is* the dominant dependence of the
  measured shape on the true shape. Forcing μ to be invariant along that direction fights the NLL
  directly and would distort the density.

And since flow #2's self-response is never read, pinning it buys nothing.

**Recommendation:** *omit* the self-response term from flow #2 (`λ_self = 0`) rather than pin it to
zero. If "set the self response in the loss to 0" meant "turn that term off", then this is already
agreed and the table entry should read *absent*. Owner's call — flagged, not acted on.

## The blend-response target — catalogue reality

The obvious source is the neighbour-only-sheared leg (19.6% of `det_meas_ngmix_g0.05_val.feather`),
where the self-response is zero by construction. **It is unusable: `measured_ngmix_g1/g2` and
`measured_galsim_g1/g2` are 0% finite in every leg where the primary is unsheared** — ngmix was
never run there. Only the superseded SExtractor moments survive.

Workaround, already coded and validated in `scripts/eval_rblend_gap.py`: use the **both-sheared** leg
and project the two-leg measured-shape difference onto the **neighbour's** shear direction. The
primary's and neighbour's shear directions are independent there (measured mean cos = −0.0000,
sd = 0.7071), so the self-response averages away:

```
R_blend_truth = ((e1_both − e1_0)·ĝ1_s + (e2_both − e2_0)·ĝ2_s) / |g_s|
```

Same projection `build_halfsim_flow_catalogue.py` already does for `delta_et1`, onto ĝ_s instead of
ĝ_p. The 45° rotated null test passes at 0.6σ.

**Cost:** the leftover self-response term is `R_self·g_p·cos2Δ`, ~20× the blend signal per row. Zero
mean, so no bias — but the label is far noisier than `delta_et1`. Rough arithmetic on 4.8M rows:
~100 bins ⇒ ~48k rows/bin ⇒ noise sem ≈ 5e-5 against a signal ≈ 6e-4, so S/N ≈ 12 per bin. Workable;
the bin design is not free. Per-row subtraction across legs is *not* available — each
(case, input_index) appears in exactly one leg of the sheared file.

## OPEN: does the sim shear positions, or only shapes?

`shifted_feature_frame` shears intrinsic shapes only; the separation vector is never moved. If the
sim shears the whole scene, part of the true blend response arrives through geometry, and a loss
that shifts only shapes would force the entire response through the shape channel — fitting the
training bins and generalising wrong. Now load-bearing, because the separation vector is in the
spec.

Test: `scripts/eval_pair_angle.py` (job `jobs/job_pair_angle.sh`, run 15328296) compares the
primary→neighbour separation for the same (case, input_index) across legs, as both the scalar
`distance` and the RA/DEC-derived vector, and regresses the fractional change on
cos 2(φ_pair − φ_γs). Identical ⇒ shape-only shear, existing machinery is faithful. Different ⇒ the
loss shift must shear the separation vector too.

## OPEN: multiplicity — training on pairs, applying to scenes

One neighbour per training row; constgold scenes have many. Today `R_blend = Σ_j` over the aperture,
and that additivity is an approximation *inside a correction term*. If the flow generates it, the
mean-additive assumption is promoted to a **correctness condition on the likelihood**.

`u` is already additive (`u = u_self + Σ_j u_j` exactly, since `v·∇log p₀` and `∇·v` both split by
scene coordinate) — handled by a node bank of variable-multiplicity scenes, non-issue. `w` is the
problem: `w_k ∝ p_flow(x̂_i | scene_k)` needs one density for the whole scene and densities don't sum.
Reconstruct through the mean, which `ConditionalMeanFlow` is already shaped for:

```
μ(x, n_1..n_J) = μ_self(x) + Σ_j δμ(x, n_j)
```

If a single explicit neighbour proves insufficient, the fallback is a spin-2 aggregate that keeps the
phase and stays fixed-size regardless of J:

```
B = Σ_j R_blend,j · exp(2i φ_j)        φ_j = pair position angle
```

Exactly linear, permutation-invariant, spin-2 (so `S_{±δ}` moves it), exactly sufficient at O(γ).
Two extra columns on the **tabular** flow — so this does *not* reopen the DeepSets trunk retired in
`Gold-V2.md` §3 Stage 1.

**Acceptance test for the whole idea:** multiplicity-binned held-out NLL — train `δμ` on pairs,
evaluate on crowded scenes binned by `J`. Flat in `J` ⇒ the sum reconstruction is valid.

## What this does and does not fix

Honest accounting on the −41%: flow #2 would be learning the same per-pair quantity BlendEMU fails
on, so improvement is not automatic. Two reasons to expect it anyway, and one reason for caution.

**Caution first:** a regression that omits a variable still converges to the conditional mean over
that variable. `E[R|scalars]` *is* the orientation-average. So orientation-blindness alone costs
BlendEMU scatter, **not** a biased mean, and cannot by itself explain −41%. (This retracts the
missing-orientation framing in WORKLOG 2026-07-28l, which overstated the case.)

**The live mechanism** is orientation-blindness *combined with* close-pair **detection selection**.
BlendEMU sees neither the pair angle nor the primary's shape, so the relative orientation of the
primary to the pair axis is entirely absent from it. Our truth sample at sub-arcsecond separation is
conditioned on both objects being detected — and at those separations detection depends on exactly
that relative orientation. The population we average over is therefore not the one the emulator was
trained on, which *does* bias a conditional mean.

Flow #2 has all three channels the emulator lacks (pair angle, primary shape, neighbour shape) and
trains on our own detected population with our own estimator, so it is the right shape of fix.

**Structural gain, independent of the −41%:** the external term disappears and a type error goes
with it. `R = Cov(ê, s)` is linear in the score and splits (INFERENCE.md 3.2); `I = Var(s)` is
quadratic and does not — a mean-response emulator cannot supply the cross term at all, so
`I + ΣR_blend` is a type error. This is the durable reason for V3.

## Risks

- **Position shear is unmodeled** — now the first open question above, with a test in flight.
- **Linearity is BlendEMU's assumption, not a theorem** — breaks once neighbours overlap. Testable
  by adding a second moment and watching NLL.
- **`I` stays optimistic** unless the flow also learns the blending broadening, not just the mean
  shift.
- **Blend-label noise** (~20×) constrains how finely the blend response can be resolved.

## First tests, in order

1. **Position shear** — job 15328296, in flight. Decides whether the loss shift needs geometry.
2. **Spin-2 signal in the pair angle** — same job. Does the truth vary with the pair angle relative
   to ĝ_s, and does the −41% survive averaging over it? Flat residual ⇒ the selection story above,
   not an omitted-variable story.
3. **Primary-shape versus pair-axis angle** — the third missing channel; one more column on the same
   extraction.
4. **Does one neighbour carry it?** From BlendEMU, the per-galaxy distribution of
   `R_blend,1 / Σ_j R_blend,j`. The tail decides whether the explicit-neighbour spec suffices.
5. **Multiplicity-binned held-out NLL** — the acceptance test (above).

Build acceptance is INFERENCE.md §3's diagnostic: `Cov(ê, s_nbr) ≈ N⁻¹Σ_i ê_i s_i^(nbr)`,
`σ ≈ std(ê·s_nbr)/√N` — consistent with zero today, should climb to BlendEMU's `R_blend` once the
spin-2 channel is in.

## Pointers

`scripts/train_joint_forward.py` (`primary_only`, `shifted_feature_frame`, `_flow_mu_shifts`,
`theta_coupling_residual`, `NEIGHBOR_FEATURES` — which already carries neighbour shape, separation
and `relative_position_angle_cos2/sin2`) · `scripts/train_measurement_model_swa_s1_truecond.py`
(the certified tabular flow #1) · `sbs_shear/measurement_model.py` (conditioning sets) ·
`scripts/eval_rblend_gap.py` (blend truth + emulator scoring) · `scripts/eval_pair_angle.py`
(position shear + spin-2 tests) · INFERENCE.md §3, §5B, §5C.3–4.

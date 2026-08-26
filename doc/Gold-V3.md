# Gold-V3 — Folding R_blend into the flow (IDEA)

**Status:** IDEA, 2026-07-28. Nothing built. Builds on `Gold-V2.md` (R_blend deliberately external)
and `INFERENCE.md` (§3, §5B, §5C.3). **Not the V3 release of `MILESTONE.md`** — that V3 keeps
R_blend external (V2.2 flow + emulator); this document is the fold-R_blend-into-the-flow idea.
Implementation paths below predate the library restructure and now live under `archive/pre-v3/`.

## The idea

Make the flow generate its own blend response, so the external BlendEMU term disappears and the
likelihood is complete. Multiplicity is the obstacle — the flow sees one neighbour, real scenes have
many — and the resolution is that **the sum over neighbours belongs in the conditioning feature and the
node bank, never in the density.**

## Why the score estimator needs this

V2 shears with `primary_only=True` so the flow learns `R_self` alone. Fine for Direction A, where
`R = R_self + R_blend` is additive. Not enough for Direction B (`γ̂ = Σs_i/ΣI_i`), which has no `R` slot:

- `R = Cov(ê, s)` is **linear** in the score, so it splits (INFERENCE.md 3.2).
- `I = Var(s)` is **quadratic**: `I = Var(s_self) + Var(s_nbr) + 2Cov(s_self, s_nbr)`. The cross term is
  the blending degeneracy, and a mean-response emulator cannot supply it.

So `I + ΣR_blend` is a type error. Either inject inside the density (INFERENCE.md §5C.3) or make the
flow generate it. V3 is the second.

## The blocker: conditioning is spin-0

All current conditioning is scalar (`nbr_flux_near/far/max`, `r_blend`), so at γ=0 the likelihood is
invariant under rotating a neighbour about the primary ⇒ `Cov(ê, s_nbr) ≡ 0` identically (INFERENCE.md
§3). **No loss term can fix this** — a response penalty cannot force a response through a channel the
architecture symmetrises away; it will distort `R_self` instead.

(Separate from `r_blend`-as-a-feature, which modulates the primary's *own* response with crowding. That
lives in `R_self` and works as intended.)

## The construction

**`u` is already additive.** `u = u_self + Σ_j u_j` exactly, since `v·∇log p₀` and `∇·v` both split by
scene coordinate. Handled by a node bank of variable-multiplicity scenes. Non-issue.

**`w` is the problem.** `w_k ∝ p_flow(x̂_i | scene_k)` needs one density for the whole scene, and
densities don't sum. Reconstruct through the mean, which `ConditionalMeanFlow` is already shaped for:

```
μ(x, n_1..n_J) = μ_self(x) + Σ_j δμ(x, n_j)
```

Train `δμ` on pairs, sum at evaluation. Same additivity BlendEMU already assumes.

**Sum in a spin-2 feature.** `r_blend` is already `Σ_j R_blend,j`, but projected onto its magnitude, and
a magnitude doesn't move under shear. Keep the phase:

```
B = Σ_j R_blend,j · exp(2i φ_j)        φ_j = pair position angle
```

Exactly linear, fixed size regardless of `J`, permutation-invariant, spin-2 (so `S_{±δ}` moves it), and
exactly sufficient at O(γ). Two extra columns on the **tabular** flow — so this does *not* reopen the
DeepSets trunk retired in `Gold-V2.md` §3 Stage 1 / cont.146.

**Loss.** One new term mirroring what exists: `λ_b · L_blend_resp` with a `neighbours_only` shift, the
mirror of `primary_only`, supervised like `theta_coupling_residual` already pins a spin-2 coupling.

## What changes about the requirement

Under Direction B, Bartlett makes a correctly specified `p_flow` unbiased automatically — there is no
R_blend number to hit. But the mean-additive assumption is promoted from an approximation inside a
correction term to a **correctness condition on the likelihood**.

## First test (no training needed)

1. From BlendEMU, the per-galaxy distribution of `R_blend,1 / Σ_j R_blend,j` — does one neighbour carry
   it? Report the distribution; the tail decides.
2. **Multiplicity-binned held-out NLL:** train `δμ` on pairs, evaluate on crowded scenes binned by `J`.
   Flat in `J` ⇒ the sum reconstruction is valid. This is the acceptance test for the whole idea.

Build acceptance is INFERENCE.md §3's diagnostic: `Cov(ê, s_nbr) ≈ N⁻¹Σ_i ê_i s_i^(nbr)`, `σ ≈
std(ê·s_nbr)/√N` — consistent with zero today, should climb to BlendEMU's `R_blend` once `B` is in.

## Risks

- **Position shear is unmodeled.** V2's `S_{±δ}` shears intrinsic shapes; blend response also depends on
  the separation vector, which shear moves. V3 forces this.
- **Linearity is BlendEMU's assumption, not a theorem** — breaks once neighbours overlap. Testable by
  adding a second moment and watching NLL.
- **`I` stays optimistic** unless the flow also learns the blending broadening, not just the mean shift.

## Pointers

`scripts/train_joint_forward.py` (`primary_only`, `_flow_mu_shifts`, `theta_coupling_residual`) ·
`sbsi/measurement_model.py` (conditioning sets) · INFERENCE.md §3, §5B, §5C.3–4.

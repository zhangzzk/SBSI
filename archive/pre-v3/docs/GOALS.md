# SBSI — Current Framing (goals · math · caveats)

Canonical statement of what we are building. Supersedes the goal framing in
`SBI_shear.md`/`SBI_shear_response.md` and the "three goals" split in memory.
Deep background stays in those docs; this is the core. (Locked 2026-07-22.)

## Goal

One forward model of measurement **and** detection, conditioned on **true**
properties; calibrate shear from it. Deliverable: overall **|m| ≤ 0.3%** on the
true-property-selected population, with the per-bin response resolved finely enough
that reweighting to a different galaxy prior stays stable. Per-bin bias need **not**
be sub-percent — only the population-marginalized m and its prior-robustness.

## Model (factored; shear is never an input — the response lives in the loss)

- **Detection classifier** `P(detected | true props, blend)`
- **Measurement flow** `P(ê₁, ê₂, meas mag, meas size | true props, blend, detected=1)`

Conditions: true shape, true mag, true size, sérsic, neighbour/blend features.
Targets: measured shape **+ measured mag + measured size**. Moving mag/size from
*input* to *output* removes the errors-in-variables floor a measured-conditioned
model has on true-property cuts (the disease diagnosed 2026-07-22).

Shear is applied by shearing the **primary's** true shape; a response term in the
loss pins the induced derivatives to matched-pair targets:
`R_shape` (shape), `R_θ` (meas mag/size → the selection term), `R_detect` (detection).
`R_blend` (neighbour-shear leakage) stays the separate per-pair BlendEMU emulator.

## Math — the response decomposition (three-fold calibration)

The estimator averages the selection-weighted shape `w·ê`. Product rule:

  ∂⟨wê⟩/∂g = ⟨w·R_shape⟩ + ⟨ê·(∂w/∂θ̂)·R_θ⟩

- **Term 1** `⟨w·R_shape⟩` — shape response of the selected sample → the flow.
- **Term 2** `⟨ê·(∂w/∂θ̂)·R_θ⟩` — selection response. A hard cut `w=Θ(θ̂−t)` gives
  `∂w/∂θ̂ = ±δ(θ̂−t)`, i.e. a **boundary** term ∝ `p(θ̂=t)·⟨ê|θ̂=t⟩·R_θ`. **Detection**
  is the special case where θ̂ is the detection proxy.

Estimate Term 2 with the **Sheldon–Huff** finite difference (re-select under ±g,
difference the mean shape): `R_sel ≈ [⟨ê⟩_{S(g+Δ)} − ⟨ê⟩_{S(g−Δ)}]/2Δ`.

Key consequence: for a **true-property** cut `R_θ = ∂θ̂/∂g = 0`, so Term 2 vanishes
and calibration is purely Term 1. Term 2 only matters for measured/detection cuts.

## Data & cuts

- Train on the **half-shear** catalogue **only**. constgold is a held-out test set —
  **never trained on** (firewall).
- Training pairs are matched, **primary-only shear** → the flow learns R_self alone,
  no R_blend double-count. Independent-shear legs are used **only** to evaluate the
  selection and detection effects.
- Cut the **primary** on **true** properties (e.g. Re > 0.3, mag < 26); keep
  **neighbours full-population** (light quality cuts only — never cut on neighbours).
  Same primary cut for the flow and R_blend.

## Caveats

- **Boundary effect.** R_sel needs both sides of the cut; same-cuts (true-property)
  training does not populate the sub-threshold side, so the flow's R_θ there is
  extrapolated. R_sel may have to come from the full-population half-shear legs (S&H),
  not the flow. Revisit once the heads are trained and the effect can be sized.
- **R_θ is small and subtle** (shear-response of measured mag/size) — validate it
  finite-difference-vs-sim before trusting R_sel; same for the detection gradient
  (dP/dg vs finite difference).
- Measured mag/size may carry a small neighbour-shear leakage not covered by the
  shape-only R_blend — measure it, don't assume it away.

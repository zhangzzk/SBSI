# Original joint response residual: population and component localization

## Main finding

Owner's subsequent population objection is valid: if the population and the
conditional self-response error transfer, their cancellation should transfer
too. The table below does **not** explain the excess joint error. The point
joint-minus-flow difference is0.014219, of which0.013270 comes from outside
the highlighted population. Joint response also contains neighbour effects.
The new [matched four-state experiment](../constant_split_20261105/rule.md#four-state-image-experiment)
tests those effects on the original constant scenes; its
[completed decomposition](../constant_split_20261105/findings.md#completed-matched-shear-response-decomposition)
finds a neighbour-related shortfall without establishing its microscopic cause.

The standalone flow's small average error hides a substantial cancellation.
A fixed truth population with circularized Re between0.15 and0.4 arcsec and
low close-crowding contributes +0.016225 to the standalone R11 error; the rest
contributes -0.008593. The same population contributes +0.017174 of the joint
constant-shear error +0.021850 on the four-case replay. This is a shared location
of a real conditional response failure, not an effect of substituting measured
selection masks. It overturns the inference that a small standalone average
exonerates the flow.

It does not prove that precisely79% of the joint error is caused by the flow.
The two columns below use different simulation families, shear experiments and
normalizations. They locate the failure and show cancellation; they do not form
a closed causal error budget.

## Matched truth-population comparison

Older16-cut GalSBI-dev checkpoints unchanged. Hard FLUX_RADIUS>3 pixels and
MAG_AUTO<25. Each side selects its own joint measurements, retaining shape,
size and flux correlations. Joint rows use actual usable flags with generated
selection. The flow uses its original matched usable pairs. Both columns here
report R11, with error sign simulation minus model, per
[response conventions](../../CONVENTIONS.md#6-response-estimators).

| Fixed population | Standalone flow error contribution, 40 cases | Joint error contribution, 4 cases |
|---|---:|---:|
| Re0.15–0.4 arcsec and nbr_close_100<0.1 | +0.016225 +/-0.003109 | +0.017174 +/-0.004149 |
| All other objects | -0.008593 +/-0.001646 | +0.004676 +/-0.002420 |
| Total | +0.007632 +/-0.003410 | +0.021850 +/-0.005689 |

Each column adds exactly, using its experiment's global normalization. The
population comprises48.2% of baseline-selected flow-check galaxies and47.9%
of selected constant-shear galaxies. `nbr_close_100` is log1p of the neighbour
flux ratio summed with a1-arcsec Gaussian distance weight; <0.1 means a weighted
ratio below exp(0.1)-1, approximately0.105. This is low crowding, not isolation.
Most of the joint/standalone difference comes from outside this population in
this comparison; the reason the cancellation changes is not yet established.

Source: [comparison and paired SEs](population_comparison.json),
[standalone cell report](flow_cell_report.json), [joint cell report](joint_report.json).
All SEs use10,000 paired whole-case bootstrap resamples, seed20260925, fixed
models/draws. Four joint cases are a diagnostic; neither these SEs nor the
point fraction79% represent total uncertainty or independent acceptance.

## Where the original joint residual sits

The fixed truth-size bins contribute:

| True circularized Re, arcsec | Contribution to joint Delta R +/-SE |
|---|---:|
| <0.15 | +0.000420 +/-0.002699 |
| 0.15–0.25 | +0.008968 +/-0.004622 |
| 0.25–0.4 | +0.012558 +/-0.003901 |
| 0.4–0.7 | +0.000043 +/-0.001228 |
| >=0.7 | -0.000138 +/-0.000277 |

The0.15–0.4 bins carry +0.021526 +/-0.007451, almost all of the point residual.
Simulation/model selected fractions in those bins are35.276%/35.382% and
40.495%/40.496%. A symmetric within-bin versus mass-mixture decomposition
assigns nearly all of these contributions to response differences inside bins,
not incorrect selected proportions. This is a bin-dependent statistical
decomposition, not a unique causal attribution.

Low crowding alone contributes+0.017527 +/-0.002015 to the joint error, versus
+0.013954 +/-0.002995 in the standalone flow. The moderate0.1–0.3 crowding bin
is particularly informative: standalone contribution-0.009090 +/-0.001263,
joint contribution+0.004804 +/-0.002072. These have different component content;
their difference cannot be assigned to emulator error without a same-population
neighbour-response test.

Magnitude<25 alone gives joint error-0.000253 +/-0.004458; adding radius>3
gives+0.021850 +/-0.005689. Radius>3 alone gives+0.023012 +/-0.005160. The
affected response is strongly tied to the radius cut.

Earlier independent training-family diagnostics found an overly flat relation
between measured-size fluctuations and shape alignment for near-PSF-size
galaxies, while mean radius transport was reproduced. That is a specific
mechanism consistent with this population localization, but the current replay
does not prove that correcting that correlation closes the constant-shear gap.
See the preserved [shape-size investigation](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-04--why-the-flows-response-fails-around-3-px-it-under-models-how-measured-shape-and-measured-size-share-noise-for-psf-sized-galaxies).

## Missing neighbour-induced selection term

### Standalone intrinsic-small-galaxy comparison

For intrinsic circularized Re0.15–0.4 arcsec, with measured radius>3 pixels
and MAG_AUTO<25, the flow's contribution to the full-sample R11 error is
+0.010572 +/-0.003621; its halftrace contribution is+0.012887 +/-0.002464.
This uses all crowding levels, not the narrower low-crowding subset above.
It represents75.83% of baseline-selected flow-check galaxies. The whole-sample
halftrace error is only+0.007987 +/-0.002744 because other intrinsic-size bins
contribute with the opposite sign.

On the emulator's standalone40-case development check, the same size range
represents75.76% of represented baseline-selected primaries. Its globally
normalized contributions, already summed over neighbour pairs per primary, are:

| Emulator term | Absolute response contribution +/-SE |
|---|---:|
| Baseline-selected shape error | -0.001793 +/-0.002036 |
| Missing neighbour-induced selection response | +0.003478 +/-0.000511 |
| Full selected-response error | +0.001686 +/-0.002083 |

Thus the intrinsic-small subset shows a positive flow response deficit; the
standalone emulator shows a nonzero omitted selection term but no significant
net selected-response error. These are contributions to each experiment's
global mean, not errors normalized within the intrinsic-size bin or relative m.
Flow uses40 held-out training-family cases; emulator uses40 development cases
with h0.2 forward independently directed half-neighbour shears. Do not add these
to claim closure of the constant-shear budget. Source:
[comparison report](standalone_size_comparison.json). Reducer17753827 uses only
the previous replay-verified case sums, reproduces all existing global values
and bootstrap SEs, and checks exact addition of disjoint size contributions.

The old standalone emulator check compared a baseline-selected shape difference
against a shape-only emulator. It omitted changes in radius/flux selection
caused by neighbour shear. On its same40 development cases, preserving actual
shape/selection coupling, the exact selected-influence decomposition is

`S1(e1-mu0)-S0(e0-mu0) = S0(e1-e0) + (S1-S0)(e1-mu0)`.

Project each term along each independently sheared neighbour direction, divide
by h, sum over retained pairs, and normalize by represented baseline-selected
primaries. Recompute mu0 and the denominator in every case bootstrap.

| h=0.2 neighbour test | Response or error +/-SE |
|---|---:|
| Old baseline-selected emulator error | -0.002556 +/-0.002293 |
| Omitted selection response | +0.004519 +/-0.000566 |
| Full selected-response emulator error | +0.001962 +/-0.002318 |

This establishes an omitted physical term, but not an explanation of the
full+0.02185 joint residual. Only half of scene neighbours are sheared, their
directions are independent, the objects have common usability, and h=0.2.
No empirical rescaling or insertion into the coherent-shear budget was made.
Source: [neighbour selection report](neighbour_report.json).

## Finite-step and component checks

The same scenes also have primary measurements at h=0.05. On identical pairs
and objects usable in all three legs, with shared baseline geometry and model:

| Quantity | h=0.2 | h=0.05 |
|---|---:|---:|
| Baseline shape error | -0.002454 +/-0.002282 | -0.003117 +/-0.008852 |
| Selection response | +0.004611 +/-0.000582 | +0.005056 +/-0.001635 |
| Total selected error | +0.002156 +/-0.002313 | +0.001939 +/-0.009307 |

Paired change in total response, h0.05 minus h0.2, is-0.000218 +/-0.008567.
No clear change is detected; the uncertainty is too large to rule out a
material finite-step effect. [Matched-step report](neighbour_step_report.json).

The standalone hard-cut R11/R22 errors are+0.007632 +/-0.003410 and
+0.008343 +/-0.003868. Half-trace averaging did not hide opposite diagonal
errors. [Component report](flow_components.json). Training and constant-shear
measurement configs use the same centred65-pixel stamps and LSST-i conditions,
as recorded in the [simulation work log](../../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-02--galsbi-full-re-simulation-launched-140-constant-shear-and-200-training-cases-on-500-cores).

## Validation and remaining boundary

- Neighbour job17750063: completed0:0 in8m30s; all40 original target-cut counts,
  label sums and model sums replayed before interpretation.
- Joint array17750068: four tasks completed0:0 in3m12s–3m14s each, at most two
  GPUs; all four cut totals replayed. Reducer17750080 completed0:0 in3s and
  verified each cell decomposition sums to the original joint residual.
- Step comparison17750106: completed0:0 in8m38s; identities and shear scaling
  verified, same denominators and predictions for both steps; dropped rows logged.
- Flow job17750130: completed0:0 in1m45s on one16GB GPU, with original40 cases
  and16 draws. Cell sums reproduce global sums; original soft/hard response
  values replay. Reducer17750131 completed0:0 in1s. The initially queued24GB
  job17750127 and its dependent reducer17750128 were cancelled before starting
  and replaced by the available16GB allocation.

No training, image simulations, checkpoint or production-code changes. No
stopped single-cut run was restarted. These tests identify the population
carrying most of the joint discrepancy and a hidden conditional flow error,
plus a smaller omitted neighbour-selection term. A causal split of the
remaining joint residual requires matching self- and neighbour-response
experiments on the same constant-shear population; it is not obtained by
adding the standalone averages. No further jobs remain.

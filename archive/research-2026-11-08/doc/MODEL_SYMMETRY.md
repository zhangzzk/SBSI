# Model symmetry candidate

## Scope and baseline — October 26, 2026

Owner requested a copy of the latest branch and an explanation of how to make
the model symmetric. Branch `codex/model-symmetry` starts from
`sampler-recipe` commit `d7d3a4015818bfa04c860cda1f6483609e8e56e1`,
in the separate `model-symmetry` worktree. The joint-kernel average below
remains a proposed reference construction. The owner subsequently requested a
single-evaluation flow with both component sign symmetries and unchanged C8.
That architecture is implemented and its retraining is launched; see
[symmetric flow training](SYMMETRIC_FLOW_TRAINING.md#request-and-model-boundary--october-26-2026).
No new production preset or joint averaging wrapper has been installed.

The relevant evidence is the [image-measurement mirror test](WORKLOG.md#2026-10-26--mirror-test-the-image-sim-c2-comes-from-the-models):
99,932 ConstGold galaxies and their reflected measurements give a reflection-even
inferred g2 component +0.002560 +/- 0.000186 (reported 13.8 sigma).
The reflection-odd component is +0.00150 +/- 0.00125.
This supports a model/pipeline symmetry problem on this population; it is not
a universal offset to subtract. The experiment does not isolate individual
components, and approximately 8.3% of original objects retain log-Z R-hat >= 1.1.

Preserve the [V3.9 identity](V39_MODEL.md#decision-and-identity) and its pinned
artifacts. A wrapper changes the probability model even with identical weights
and must have its own identity and cache provenance.

## Recommended first candidate: exact reflection of the joint kernel

Let x denote the latent scene, g the two-component shear, and
y = (e1, e2, radius, flux) a measured outcome. Define reflection M by

- x: negate the raw target e2 input; reflect any directional scene quantities
  before feature construction when working with full scenes.
- y: negate measured e2; retain radius, flux, e1, and radial log-gap.
- g: (g1, g2) -> (g1, -g2).

For the current compressed F8/C8 inputs, only `e2_input_p` changes. Scalar
crowding, circularized radius, magnitudes, and Sersic index are invariant.
Reflection acts on physical values before the checkpoint preprocessors.
Negating a standardized coordinate is generally wrong when its mean is nonzero.

Write the usable-event kernel, including disk transport, as

```text
K(y | x, g) = q(x, g) f_disk(y | x, g)
K_M(y | x, g) = K(My | Mx, Mg)
K_sym(y | x, g) = [K(y | x, g) + K_M(y | x, g)] / 2.
```

Here q is C8's usable-event probability; f_disk is the normalized joint
four-output F8 density after the existing response transport. Reflection has
absolute Jacobian one. Applying M again exchanges the summands, proving

```text
K_sym(My | Mx, Mg) = K_sym(y | x, g).
```

The implementation should evaluate both complete log kernels and combine them
with logaddexp minus log(2). Average densities, not log densities, predicted
means, observed ellipticities, or final shear estimates. No fitted offset enters.

## Classifier, conditional flow, and exact generation

Integrating over all measured outcomes gives

```text
q_sym(x, g) = [q(x, g) + q(Mx, Mg)] / 2
f_sym(y | x, g, usable) =
    [q(x, g) f_disk(y | x, g)
     + q(Mx, Mg) f_disk(My | Mx, Mg)]
    / [q(x, g) + q(Mx, Mg)].
```

If both q values vanish, the usable kernel is zero and no conditional usable
draw is requested. Otherwise the conditional mixture weights are proportional
to the two q values; they are not generally one half.

An exact parent-level generator is simpler:

1. Choose the original or reflected branch with probability one half.
2. Evaluate that branch's C8 probability and draw the usable event.
3. If usable, draw its complete F8 outcome and apply its disk transport at the
   branch shear, using that same draw's radius and flux for the response.
4. Reflect the measured shape back when the reflected branch was chosen.
5. Apply the measured selection cuts to the resulting outcome.

For weighted selection integration, average the two branch contributions
q times the cut indicator instead of drawing the Bernoulli event. Common
random numbers may couple branches and shear nodes, but each branch must
retain the correct marginal distribution. For this wrapper, exact paired
samples can be obtained by exchanging the branch and reflecting the output;
the base F8 flow need not itself map a sign-flipped Gaussian draw equivariantly.

Separately averaging q and f is also capable of defining a symmetric model,
but their product includes cross terms pairing one branch's classifier with
the other branch's flow. It is a different model from the joint average above.
Use the joint construction as the first candidate to preserve their pairing.

## Disk response and selection normalization

The retained emulator takes scalar sizes, fluxes, Sersic indices, separation,
and measured radius/flux. Its pair sum b is invariant under reflection.
The velocity b*g reflects with g, and the radial-tanh/Mobius transport satisfies

```text
T_(b*Mg)(Me) = M T_(b*g)(e).
```

Thus the existing disk response needs no retraining for reflection symmetry,
provided its actual feature construction, transport, and density Jacobian
pass a paired numerical check. This is a structural conclusion from
`scene_model_inputs`, `response_pairs`, and `disk_response_transport`;
the October 26 experiment alone did not establish it.

For the prior p(x) and measured-cut indicator C(y), use the same kernel in

```text
Z_sym(g) = integral p(x) C(y) K_sym(y | x, g) dx dy.
```

If the prior and cuts are reflection invariant, changing variables gives
Z_sym(g1,g2) = Z_sym(g1,-g2). Its g2 derivative is then zero at g2 = 0
where the derivative exists. This must follow from consistent integration;
do not manually erase the old selection slope. The current radius/flux and
absolute-ellipticity cuts are invariant. Verify the numerical prior contract
and use paired integration to avoid introducing a finite-prior asymmetry.

The old asymmetric selection slopes, posterior chains, and likelihood caches
do not become symmetric merely by relabelling them. Produce new candidate
receipts; any reuse requires a demonstrated equivalence for the new target.

## Broader symmetry: square-image group before continuous rotations

The two-branch construction addresses reflection in e2 exactly. It alone
does not enforce e1 sign symmetry or arbitrary rotation equivariance.

For both ellipticity components, one possible extension is the group of
four sign actions (e1,e2) -> (+e1,+e2), (+e1,-e2), (-e1,+e2), (-e1,-e2),
applied to input shape, output shape, and shear together. These are the four
distinct actions of square-image rotations/reflections on spin-2 shape.
The current model retains no neighbour direction or anisotropic PSF feature,
so the other input features remain unchanged. Full scenes still require
the corresponding geometric transformations.

For any chosen finite group G, replace the half sum with
K_G = sum_A K(Ay | Ax, Ag) / |G| and use q-weighted conditional sampling.
This gives exact symmetry under G. Four branches require four base-kernel
evaluations. Two branches require two; actual sampler wall time must be
benchmarked because scene proposals and response work can be shared.

Do not equate this discrete symmetry with all-angle rotation symmetry.
A square pixel grid can retain orientation-dependent measurement behavior.
Continuous rotation averaging would require its own physical justification
and either an exact equivariant model or a controlled angular integration.
No continuous-rotation assumption is needed to test the observed mirror defect.

## Implementation boundaries and validation order

The first implementation should expose an explicit joint-model adapter with
consistent kernel, generation, and selection entry points. The current
`SummaryLikelihood.log_kernel` is a useful reference for the two complete
kernel evaluations. Wrapping only `MeasurementModelBundle.sample` is
insufficient: density paths call the underlying model directly, including
`physical_log_prob_with_gap`.

Before a population run:

1. Check raw-coordinate reflection, M squared = identity, invariant auxiliary
   outputs/log-gap, and transport plus Jacobian equivariance, including the
   disk-edge precision path.
2. Check C8 probability invariance and joint density equivariance on fixed
   scenes and measured outcomes at zero and nonzero shears. Include an
   intentionally asymmetric toy model with unequal q values to catch
   incorrect mixture weights and generation/density mismatches.
3. Check candidate sampling against its evaluated density and verify that
   selection uses the same usable-event model and cut order. Include fresh
   prior scenes and common-random-number shear stencils.
4. Check that the sampler, initialization bank, co-moving scores and caches
   all use the candidate kernel; test the reflected shear-score transformation
   and report convergence failures.
5. Freeze the candidate and numerical tolerances before comparing it on
   model mocks and image measurements. Repeat the original/mirrored
   image-measurement diagnostic, with paired errors and unchanged cohort
   accounting. ConstGold remains test-only, not a hyperparameter selector.

Symmetry guarantees reflection equivariance of the exact model. It does not
force a noisy data set's g2 estimate to zero, ensure sampler convergence, fix
population mismatch, or establish multiplicative calibration. The image
test must measure the change in the reflection-even component rather than
subtracting the previously observed +0.00256.

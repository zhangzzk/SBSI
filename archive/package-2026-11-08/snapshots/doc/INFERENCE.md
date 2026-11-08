# Inference mathematics and implementation

## Current identities

`infer-1.0-galsbi` names the continuous GalSBI-prior inference core.
`like-1.0-galsbi` names the current model set. These identities are separate;
each scientific run must also pin the sampler settings, observing conditions,
selection, input population, seeds and source. See the
[current implementation](CURRENT_INFERENCE.md#entry-points).

## Selected likelihood

For a latent scene z drawn from the declared population prior P, let q(z,g)
be its usable-measurement probability and f(y|z,g,U) the joint density of
measured shapes, radius and flux, including the declared neighbour transport.
For an observed y that passes the measured-output selection S:

```text
L(y,g) = integral q(z,g) f(y|z,g,U) dP(z)
A(g)   = integral q(z,g) Pr(Y in S | z,g,U) dP(z)
p(y | selected,g) = L(y,g) / A(g).
```

The classifier predicts usability, not membership in S. Radius and flux
remain joint flow outputs because selection and the disk response depend on
them. Each shear leg uses its own selected-population normalization. Forward
population draws estimate A independently from per-observation posterior
integration. Their uncertainty must be propagated.

## Continuous-scene integration

The current sampler targets the scene posterior at a declared reference
shear, with target type marginalized and an explicit neighbour scene. MH
updates and optional tempering integrate the scene uncertainty. Nearby shear
likelihood ratios use co-moving coordinates with the required prior ratio and
Jacobian. Proposals are frozen after warmup; common random streams and the
original numerical evaluation order are preserved.

At a fixed observed radius/flux pair, the implemented likelihood compresses
a scene to eight galaxy/crowding inputs plus one summed disk coefficient.
The coefficient is evaluated at that observed photometry. This is an
observation-dependent deterministic summary, not a new fitted prior density.
The multistage flow needs a second coefficient at reference photometry, so
it requires a separately validated adapter.

## Score and information

For each observation, posterior integration estimates the shear score and
curvature contributions. Subtract the derivative of log A(g) from the score
and include its curvature in the information. The score/information solve
updates the expansion shear. The historical reference mock used a bright-only
second evaluation and retained its C1 information; those settings are part of
that run, not implied by the core pipeline name.

The Bartlett identity, selected-population derivatives, and the Eulerian
and Lagrangian forms are derived in [MATH.md](MATH.md#1-the-model).
Definitions of shape, selection, response, pairing and m are in
[CONVENTIONS.md](CONVENTIONS.md#6-response-estimators).

## Finite-catalogue compatibility

For discrete prior atoms, replace each integral by a weighted sum over atoms.
The retained `v1.1-infer` and `v1.2-infer` configurations select different
importance-sampling estimators; `configs/likelihood.json` still names the
historical `v3.2-like` likelihood. Naming the GalSBI core does not alter those
defaults. See the [finite-prior contract](CATALOGUE_PRIOR.md) and
[archived commands and estimator settings](../archive/research-2026-11-08/snapshots/doc/INFERENCE.md#numerical-estimators).

## Evidence boundary

Loading a model and replaying its density do not validate an inference result.
The historical paired mock narrowly failed its joint two-sigma gate; a full
uniform-recipe truth test and end-to-end validation of the current fiducial
remain incomplete. See [the evidence summary](HISTORY.md#reference-mock-and-limits).

# N19 analytic state-dependent phase control

## Predeclared candidate and scope

Before inspecting any new outcome, fix epsilon = **0.1**, r² = e1²+e2²,
F = (1+r²)/(r²+0.01), a = (-F e2, F e1), a_prime = (-F e1, -F e2).
The two-vector control is C = a T + a_prime, namely
C1 = -F(e2 T + e1), C2 = F(e1 T - e2). The candidate score is **S - C**.
No coefficient fit, empirical recentering, regularizer search or sign choice
is permitted after results. This is Arbor n19 under n15, not a sampler or
heldout change. The external declaration is
`analytic_phase_n19_20261022_declaration.json` in the sampler-search data root.

Use only the four v1 pilots and eight each of v5/v6v100/v7, their successful
phase-v2 arrays and saved intrinsic summaries. Keep every particle, stride 1,
original Zhat and full N denominators. Compare original score, the existing
frozen constant control, and this fixed analytic control. The constant-control
comparison on its own v1 pilot data is explicitly in-sample/descriptive.
Raw F32-sampled and exp(Delta logK)-reweighted F64 measures stay separate;
no exact zero expectation is asserted for the rounded F32 sampling program.
All original score/normalizer gates remain unchanged. CPU Slurm only: no
new model forward pass, GPU, sampling, heldout input or higher-effort-arm input.

## Derivation to verify against actual code

At fixed radius and other phase-invariant latent variables, phase has a
uniform native prior, and T = derivative_psi log K. Since a and K are
periodic, integral (a T + derivative_psi a) K dpsi = integral derivative_psi
(a K) dpsi = 0. F depends only on radius, so derivative_psi a = (-F e1,-F e2).
The epsilon regularizer makes a and a_prime continuous and zero at e=0;
no division by radius or phase convention at the origin is required.

The actual reduced-shear map is e'=(e+g)/(1+conjugate(g)e), with real spin-2
components. Its g=0 Jacobian has rows
[1-e1²+e2², -2e1e2] and [-2e1e2, 1+e1²-e2²]. Dotting each shear column with
(-e2,e1)/r² gives the unregularized phase tangent
(1+r²)(-e2,e1)/r². The chosen coefficient shrinks it by r²/(r²+0.01).
This motivation is specific to the intrinsic shear map at g=0; the full
score also has observed-shape response/transport terms. The zero-expectation
periodic product-rule identity does not require claiming that C removes the
entire shear score, or that the same tangent expression holds at nonzero g.

The planned independent tests check this Jacobian and its radial/tangential
decomposition against finite differences of `apply_shear_to_ellipticity`,
nonuniform periodic integration, product-rule signs, origin/wrap behaviour,
and an intentionally missing a_prime term. Evaluation follows only after
those tests, with frozen code/source/input hashes and no outcome selection.

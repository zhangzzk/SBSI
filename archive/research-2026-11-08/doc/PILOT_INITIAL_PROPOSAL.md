# Frozen pilot initial proposal

## Predeclared n13 rule

This rule was written before fitting any pilot moments. Only completed
`dev_v1_0`, `dev_v1_1`, `dev_v1_2`, and `dev_v1_3` in
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021`
are authorized fit inputs. Later development populations and held-out
observations do not enter this fit. The hypothesis is that a normalized,
independent-pilot initial mixture with informed phase reduces normalizer
concentration at the same effort. Mathematical validity does not establish
that empirical claim.

For each observation, retain the original two measurement-based means and
shape scales, but explicitly construct **Student-t(df=5)** components, with
uniform phases and 20% total mass in their original relative weights. Do not
copy the pilots' Gaussian family. Add one Student-t(df=5) component per pilot,
each with 20% mass. Each component has its pilot's coordinate means in the
first nine coordinates. Use empirical variance (divisor N); convert marginal
variance to t shape by multiplying by (5-2)/5. Inflate the empirical standard
deviation by 1.5 and floor it at 0.05 times the maximum original measurement
coordinate scale in that coordinate, **before converting to t shape**.
The original coordinate scale is the original Gaussian SD, numerically equal
to the retained broad t shape SD; it is not the broader t marginal SD.
Pilot shape SD is exactly
`sqrt(3/5)*max(1.5*sqrt(var), .05*max(original_coordinate_scale))`.

The phase centre is the per-pilot circular mean. With empirical resultant R,
set the desired inflated resultant to `R**(1.5**2)` and solve
`I1(kappa)/I0(kappa) = desired_resultant`, capped at kappa=10000. This is exact
1.5 inflation of circular SD `sqrt(-2*log(R))`, and approaches ordinary
standard-deviation inflation for concentrated circular laws. Resultants at
or below 64 machine epsilons use the uniform phase (kappa=0, centre=0).
R is bounded to [0,1] to handle floating-point roundoff. This is a deterministic
fit of proposal parameters, never clipping, rejecting, or redrawing samples.

Every component's real-coordinate and phase densities are multiplied before
summing the mixture. Broad uniform-phase t components preserve full support
on R^9 times the circle. Invalid native-prior draws must retain zero weights.
All fit choices are heuristic proposals, never corrections to the posterior.

## Preserved specification discrepancy

The first implementation interpreted "original broad SD" as the retained
broad t marginal SD, while the pre-fit message's intended reference was the
original measurement Gaussian SD. As a result, its active shape-scale floor
was `.05*original_coordinate_scale`, instead of the declared
`sqrt(3/5)*.05*original_coordinate_scale`. The first fit is preserved as
`pilot_initial_proposal_n13.json` (SHA256
`3a98e503b02f0832ec7499fdc69f6f8db1beef3bb85d5af049c1c20527f2f729`), with
CPU fit receipt 17419499 and snapshot `source_initial_proposal_n13`.
It is superseded and must not be used. No scientific sampler run used it.
The corrected schema v2 records the floor's reference and operation order,
and a constant-particle test forces the floor branch. The corrected artifact
uses a fresh `_v2.json` filename; no scientific outcome selected this revision.

## Two mechanisms under test

The frozen q0 changes both initialization and subsequent sampling. It seeds
the initial particle population and determines every intermediate annealing
density. The existing independence block also proposes from that same q0
throughout the schedule, including at beta=1. Its Hastings log ratio is
`beta * delta(log(p_analytic*K) - log(q0))`; the implementation already uses
the complete normalized phase-dependent q0. A better initial law may therefore
improve later independence moves as well as starting states. Any later
efficiency improvement must be attributed to the combined intervention unless
a separate experiment holds one of these mechanisms fixed.

The four pilots can share missed modes, and their phase concentrations can
reflect confinement. Retaining broad support prevents a mathematical support
loss but cannot guarantee practical discovery of rare target or environment
explanations. The environment law and mutation schedule remain unchanged.

## Independence and identity

Freeze the artifact before production; reject every pilot seed in production.
Record hashes of pilot protocols, particles, results, all original source
hashes, and the fit implementation. Verify observation order, shear, reference
and observation hashes, numerical density batch, bridge, selection, and the
native target-source files recorded by the original pilots:
`galsbi_target_coordinates.py`, `galsbi_summary.py`, `summary_gradients.py`,
and `models_v3_9_like.json`. These records do not cover every transitive import.
The particle wrapper's original hash remains in provenance; wrapper equality
is not required because the intervening changes implement mutation geometry
and extinction bookkeeping without changing the target. No new target code
is introduced by this experiment.

The artifact owns the effective proposal law; `--initial-proposal` cannot be
combined with explicit `--proposal-family`, `--proposal-df`, or `--faint-shift`.
Its normalized density enters the existing bridge and independence Hastings
correction. Fresh production populations may share the same frozen artifact;
comparison with populations whose seeds entered its fit is rejected. No
held-out accuracy claim is possible from these four fitted observations.

## Frozen artifact and validation

CPU job 17419561 produced
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/pilot_initial_proposal_n13_v2.json`,
SHA256 `7f11db32019d4e0014a10acc61ea890c17d6fc61b67d0fd9ff36afe3763c46e4`.
Its four pilot seeds are 2026102130, 2026102131, 2026102132, and 2026102133.
It applies only to observations `[58165, 0, 1, 8]` at shear `[0, 0]` with
the recorded reference and source identities. Its source snapshot is
`source_initial_proposal_n13_v2` under the same data root, containing 157
manifest-listed source/doc files and zero bytecode/cache files.

The CPU fit command, run inside that snapshot, was:

```bash
R=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -m scripts.fit_particle_initial_proposal \
  --runs "$R"/dev_v1_0 "$R"/dev_v1_1 "$R"/dev_v1_2 "$R"/dev_v1_3 \
  --output "$R"/pilot_initial_proposal_n13_v2.json
```

The CLI refuses to overwrite a frozen artifact. For an authorized production
driver run, add `--initial-proposal "$R"/pilot_initial_proposal_n13_v2.json`
and use a fresh seed. Omit `--proposal-family`, `--proposal-df`, and
`--faint-shift`; the artifact owns all effective proposal parameters.

CPU test job 17419552 passed 70 tests in 19.51 seconds across proposal,
initial-artifact, and extinction suites. These include SciPy normalized
density agreement, circular normalization and sample cross moments, default
RNG/numeric replay, forced scale-floor fitting, immutable receipt validation,
seed/identity rejection, correct bridge/Hastings use, and driver integration
with no, some, and all observations extinct. This is correctness evidence;
no scientific efficiency result is claimed for this proposal yet.

# Finite catalogue prior and caches

## Finite-scene likelihood

Let `z_j` be a scene atom with prior mass `pi_j`, `S_g` the configured
shape-only shear map, and `y_i` a retained measurement. SBSI evaluates

```text
A_i(g) = sum_j pi_j q_j(g) f_j(y_i - b_j(g) | S_g z_j, U)

b_j(g) = R_blend,j [e(S_g z_j) - e(z_j)].
```

With measured-output predicate `W`, the population normalization is

```text
B_W(g) = sum_j pi_j q_j(g)
         Integral W(y + b_j(g)) f_j(y | S_g z_j, U) dy.
```

The selected likelihood is `A_i(g)/B_W(g)`. Every observed row already passes
`W`, so selection changes the population normalization rather than adding a
second numerator indicator.

For V3.5-like, `q_j(g)=p(U_j(g)|x_j(g))` comes from the equal-probability
nine-input classifier ensemble. The measurement density remains joint in
`(e1,e2,FLUX_RADIUS,flux-from-MAG_AUTO)`, so the 25.8-magnitude and 0.75-arcsec
radius cuts are integrated over the flow. A direct `p(S|x)` is not substituted.

## Scene store

`scripts/build_scene_prior.py` writes:

- `galaxies.parquet` — truth rows and prior masses;
- `neighbours.npz` — the directed guarded neighbour graph;
- `manifest.json` — source, schema, geometry, counts, and provenance.

Only positive-mass rows are sampled as primaries. Zero-mass rows remain to
supply neighbour context. The guard radius must cover every downstream model
aperture. Under the current transform, intrinsic ellipticity changes while
flux, size, position, separation, and graph membership remain fixed.

Model views are separate because the flow, classifier, and response emulator
have different feature contracts. All views remain aligned to the same primary
atom index. Duplicate and unmatched identifiers are rejected or explicitly
dropped and reported.

Large source catalogues can be assembled, extracted, sharded, finalized, and
audited with the five active scene-prior scripts listed in `doc/INFERENCE.md`.

## V3.6-like uncut 24-million-atom subset

The 2026-09-20 inference preparation uses a uniform sample without replacement
of24,000,000 primaries from all139,936,000 rows in the200-scene source store,
seed20260920. This is a finite-prior approximation, not a truth-property cut.
Each selected atom receives global mass1/24,000,000; old cut-dependent masses
are discarded. Keep the measured radius>0.6arcsec and magnitude<25.8 selection
separate from this uncut prior.

`scripts/subsample_disk_prior.py` checks source hashes and writes sharded
galaxies, unchanged zero-flow contexts, complete per-primary response pairs,
global source identities and original random draw ranks. Neighbours need not
be sampled primaries: do not prune them or reinterpret their source IDs as
compact atom positions. `DiskCatalogueModelCache(source_atom_ids=...)` uses
those original identities when constructing classifier features. Random ranks
allow nested smaller-prior comparisons; no prior-size convergence claim has
yet been established.

`jobs/job_subsample_v36_prior.sh` writes `prior_subset24m/manifest.json` under
the20260920 V3.6 run root. This disk-specific manifest is not a legacy
`ScenePrior` store. The dedicated disk driver now implements persistence/loading
and has passed small and full-prior real-model pilots. Full-prior caches are
complete and all production observation partitions finished. Combination failed
the positive-information check and targeted curvature probes are in progress;
see the [curvature audit](WORKLOG.md#2026-09-21--diagnose-failed-v36-production-curvature).
The legacy driver's preparation-only guard
and unbound generic16k config remain intentional. See
[disk driver](INFERENCE.md#v36-like-disk-driver).

## Derived caches

Caches are reproducible performance artifacts, not independent scientific
inputs. Each manifest pins everything required to interpret its arrays.

### Model cache

`CatalogueModelCache` stores atom-aligned flow conditions, usable probabilities
and optional blend shifts for requested shear views. Reuse across shear is
allowed only for features declared invariant under the configured transform.

### Blend response

`CatalogueBlendResponse` stores one `R_blend` value per active atom. Its
identity includes the scene, response-emulator model and metadata, observing
conditions, and pair configuration. An atom with no supported neighbour pair
can have a physical zero; a missing joined key is an error.

### Proposal coordinates

`ProposalCoordinateTable` summarizes QMC flow draws in measured-output space
for candidate retrieval. It records target order, location and dispersion,
draw count, seed, flow hash, and scene identity.

### Measured-selection normalization

`CatalogueSelection` stores atom-aligned pass probabilities for an exact
predicate, shear stencil, joint flow, blend response, draw depth, and random
seed. Common random numbers are retained across shear views. Cache updates are
atomic and cannot silently replace a different predicate or upstream model.

## Importance evaluation

The small-catalogue oracle is the exact atom sum. For a large prior, v1.1 uses
the defensive proposal

```text
q_i(j) = epsilon*pi_j + (1-epsilon)*q_local,i(j).
```

Each sampled term keeps the exact correction `pi_j/q_i(j)`. The global
component gives every positive-mass atom support and bounds the importance
ratio by `1/epsilon`.

The local proposal:

1. retrieves a broad support from cached flow coordinates;
2. reranks it with flow uncertainty;
3. evaluates the exact initial-shear mass on that support;
4. mixes the normalized local mass with the defensive prior; and
5. freezes atom IDs and proposal probabilities across the full shear stencil.

Object processing is streamed. Nested draw ladders are deterministic prefixes,
so increasing the budget never changes an earlier draw or another object's
stream.

The v1.2 alternative sums the candidate stratum exactly and samples only the
complement from a tilted whole-catalogue proposal. The exact stratum enters
once; the complement retains its importance correction. See `doc/INFERENCE.md`
for the configured budgets.

## Diagnostics

Per-object draw rungs record effective sample size, largest-weight share,
relative standard error, and generalized-Pareto tail index. Undefined tail
fits are counted rather than relabeled as benign. ESS fractions are not
directly comparable between mixture and exact-stratum estimators; relative
error is the relevant common diagnostic.

## Partitioning and combination

Independent observation partitions may be combined only by summing their
score, information, count, and influence sufficient statistics before solving.
Combination refuses overlaps, gaps, mixed centres, mixed scene/model/cache
identities, different selection predicates, or different random-stream
contracts.

## Validation boundary

At minimum, validate:

- exact-versus-importance agreement where feasible;
- nested-rung and independent-seed stability;
- score centring and recovery of both shear signs;
- score-variance versus observed-information consistency;
- full two-component finite-difference cross terms;
- cache feature parity and row alignment; and
- reproducibility from a frozen manifest.

Likelihood-generated closure tests the algebra and sampler under the declared
density. Only fresh image-level closure tests the combined adequacy of the
measurement, usable-event, selection, and response models.

The complete pre-cleanup discussion is preserved in
`archive/research-2026-09-14/doc/CATALOGUE_PRIOR_FULL.md`.

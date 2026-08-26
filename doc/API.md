# SBSI API — scope, contract, and public API

Split out of `AGENTS.md` on 2026-08-18. `AGENTS.md` carries the one-line
invariants and points here for the detail; where the two disagree, `AGENTS.md`
wins.

## Scope

- Keep classifiers, conditional flows, response prediction, validation, and
  Bayesian inference in SBSI.
- BlendEMU owns image simulation, measurement, simulation-catalogue production,
  and emulator training. SBSI may call those supported APIs/CLIs but must not
  copy their rendering or measurement implementation.
- SBSI owns forward-response catalogue preparation from a user truth
  catalogue: input validation, neighbour finding, representative-neighbour
  selection (nearest by default, optionally impact-ranked), the
  one-row-per-primary flow scene view, pair construction, training-matched cuts
  and rescaling, and keyed alignment of emulator predictions.
- Do not copy SBSI inference or flow code into BlendEMU, or move BlendEMU
  catalogue builders here, unless the owner explicitly changes this boundary.
- The next emulator training/tuning update belongs in BlendEMU.

## General API contract

- Every training, validation, prior, and inference catalogue is supplied by the
  user as a DataFrame or external path. Never hide a project catalogue in an
  API default.
- Flow checkpoints, emulator artifacts, and detection-classifier checkpoints are
  explicit paths. `get_model("V3")`, `get_model("V3.1")`, `get_model("V3.2")`,
  and `get_model("V3b")` are path-only conveniences; no
  behavior may branch on those names.
- Model support is read from checkpoint metadata or supplied explicitly. Never
  infer a domain from a filename.
- Reusable behavior belongs in `sbsi/`.

## Public API

1. `sbsi flow`: config-driven conditional-flow training and explicit tuning,
   backed by the reusable implementation in `sbsi.flow`.
2. `sbsi.response`: ensemble response prediction and blend-response
   composition.
3. `sbsi.scene_prior`: reusable finite catalogue atoms and their guarded
   neighbour graph.  It exposes separate flow, detection, and response-pair
   views because those models use different apertures and neighbour rules.
4. `sbsi.catalogue_likelihood` and `sbsi.catalogue_closure`: the exact
   small-catalogue flow+detection likelihood and closure oracle, plus
   `sbsi.catalogue_sampling` for target-specific defensive importance
   sampling, and `sbsi.catalogue_blend` for a fixed atom-aligned external
   response cache.  Mock generation, likelihood residuals, and measured-cut
   draws apply the same finite-shear blend shift.
5. `sbsi.catalogue_null`: the streamed two-component `MATH.md` §5 expansion
   and its exact-oracle, sampling, information-identity, and tail gates.
6. `sbsi.inference`: the earlier shape-prior Bayesian interface.

The operational catalogue-prior command path is deliberately small:

- `configs/infer_v1.json` is the canonical **Infer V1** numerical setup. It
  versions sampling and solving independently of model presets such as V3.2;
- `configs/default_catalogue_prior.json` names the current immutable,
  sharded FS2 prior manifest; shard-aware consumers must preserve its declared
  global mixture masses;
- `scripts/build_scene_prior.py` builds the guarded reusable scene;
- `scripts/build_catalogue_blend_response.py` builds the fixed atom-aligned
  response cache;
- `scripts/prepare_image_closure_mock.py` maps one declared BlendEMU image leg
  into the exact flow outputs, with source and shear provenance;
- `scripts/run_catalogue_closure.py` generates or loads a frozen mock and runs
  the exact/importance likelihood;
- `scripts/run_section5_powered_exact.py` is the first catalogue-prior null
  gate: matched finite priors and mocks test the exhaustive sum, independent
  property banks, and numerical derivatives against exact Torch autograd;
- `scripts/run_section5_powered_importance.py` is the full-prior powered gate:
  it evaluates the local numerical expansion at `g=0` with `h=0.00125`,
  measured cuts off and `R_blend=0`; its local proposal is posterior-adapted
  using exact zero-shear likelihoods on the cached nearest-candidate support;
- `scripts/run_section5_nonzero_closure.py` is the paired local multiplicative
  test.  It injects `+-0.00125` with common mock/proposal streams, retains
  object moments, and gates the joint score-response ratio on sampler and
  response-tail convergence.  Its optional rotational ring is diagnostic-only;
- `scripts/run_section5_numerical_recenter.py` maximizes the full catalogue
  likelihood for either a saved mock or a newly generated likelihood mock. It
  supports the fixed atom-aligned `R_blend` cache and measured-output selection,
  including `P_pass(g)` in the detected-and-selected population normalization.
  It uses `h=0.001` by default, distinct from the null estimator's `h=0.00125`.
  Its default `initial_center_posterior_adapted` proposal weights the cached
  candidate support by exact `pi Pdet L(initial)`, mixes it with the defensive
  prior with exact `pi/q`, and reuses the candidate likelihood at the initial
  point. The resulting draw is held fixed for every two-component stencil and
  line-search point. `distance_kernel` remains an explicit fallback. It accepts
  only likelihood-increasing safeguarded updates. After recentering it makes
  one streamed pass at the final shear and records ESS, maximum-weight
  concentration, local/global proposal
  contributions, proposal method/reference/reuse, the complete flow-evaluation
  split, and runtime. The bandwidth result is inactive/null for the adapted
  method. Loaded image mocks must pass their kind, shape-only shear-map, model,
  target-order, injection, and
  output-hash manifest gates. Loaded likelihood mocks additionally require the
  runner's exact generation and implementation identity. This is the
  finite-shear closure path; the zero-centred one-step estimator remains the
  local/null path;
- the earlier combined-null and cross-prior reweighting experiments are
  archived under `archive/infer-v1-development/`; they are provenance, not
  production entry points;
- `scripts/summarize_catalogue_closure.py` reports individual profiles;
- `scripts/summarize_catalogue_bias.py` audits the powered zero/sign panel;
- `scripts/summarize_numerical_recenter.py` pairs independent `+/-` image legs
  by simulation-case block and reports additive, multiplicative, and
  cross-component estimates with case-block and likelihood-curvature errors.
  It re-hashes the frozen analysis tables, requires converged positive-
  information fits and complete sampler diagnostics, and refuses mixed model,
  cache, stencil, optimizer, implementation, duplicate-mock, or overlapping-
  case identities.

The detection checkpoint metadata gate accepts only the declared seven
spin-0 inputs, so the current likelihood legitimately reuses its cached
`Pdet` across shear views. This is only a software contract. ConstGold images
show a missing diagonal detection-selection response of
`-0.009191 +/- 0.000275`; the checkpoint has no ellipticity input with which to
model it. Precision image closure therefore requires a detection/usable-event
model update even after numerical sampling stabilizes.

The image-stage order is: uncut full-model closure with at least two proposal
seeds; reconciliation of the detector and usable-measurement event; then the
realistic and stress measured cuts. A measured-cut likelihood closure alone
does not authorize a selected-image closure claim.

The superseded §5B finite-M score/quadrature implementation and its tests live
under `archive/infer-v1-development/legacy-score/`. It is not installed as
`sbsi`, collected by pytest, or available as an alternative production entry
point.

The seven-point profile mode in `run_catalogue_closure.py` is retained as a
nonlinear MLE fallback. It requires a supplied shear bracket and is not the
default null-closure estimator.

Both response components are always required:

```text
R_model = R_flow + R_blend
m = R_sim / R_model - 1
```

Failed catalogue joins are rejected, never silently assigned `R_blend = 0`.
An atom with no neighbour inside the emulator's supported pair selection has a
physical zero external response and is reported separately from alignment
failures.

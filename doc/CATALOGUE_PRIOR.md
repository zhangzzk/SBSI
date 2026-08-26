# Catalogue-prior inference

## Current implementation boundary

The catalogue-prior implementation uses the measurement flow, detection
classifier, and an optional fixed external BlendEMU response.  The exact
finite sum remains the small-catalogue oracle; defensive importance sampling
is the scalable implementation for the multi-million-row scene prior.

The null-closure target is the BFD-like local expansion in `MATH.md` §5.
It evaluates the marginalized log likelihood at `(0,0)`, `(+-h,0)`, and
`(0,+-h)`.  Comparison with exact Torch autograd selects `h=0.00125`: larger
steps have visible truncation error and smaller steps lose second-derivative
accuracy to float32 cancellation. Importance atoms are fixed across every
view and the object loop is streamed, so a catalogue-sized object-by-atom
tensor is never retained. That validated null path keeps measured-output cuts
disabled and external `R_blend=0` so its historical gates remain comparable.

Finite-shear closure uses `run_section5_numerical_recenter.py`. It maximizes the
same catalogue-marginalized likelihood with a full numerical 2x2 Hessian and
now supports the complete model: detection, fixed atom-aligned `R_blend`, and a
measured-output cut with detected-and-selected normalization. Proposal atoms
and probabilities remain fixed throughout the stencil and line search. The
operational finite-shear stencil is `h=0.001`; this is separate from the
`h=0.00125` validated for the zero-centred local/null estimator. At the initial
centre, the default proposal adapts the cached nearest-candidate support with
the exact catalogue numerator `pi Pdet L(initial)`, mixes it with the defensive
global prior, and retains the exact `pi/q` correction. The initial candidate
likelihood is reused when assembling the first scalar likelihood, after which
the same draw is frozen for every stencil and line-search point. The older
distance-kernel proposal remains an explicit fallback.

The
selection integral uses the same fixed-seed QMC flow latents at every shear;
its sampling depth and independent seed must be converged separately because
the population term is multiplied by the number of observed objects. A final
streamed pass at the recovered shear reports mean/median/p10 ESS, ESS fraction,
p90 maximum-weight fraction, and the evidence fractions supplied by the
defensive global and outside-local draws. It is included in the declared flow-
evaluation count rather than hidden as reporting overhead.

The compact model cache stores zero-shear flow conditions and detection once.
For the declared checkpoint, only `e1_input_p,e2_input_p` change at trial
shear; size, magnitude, profile, and neighbour-flux inputs are shared. The
detection checkpoint feature list is checked against an explicit spin-0
allow-list before inference starts. Nonzero views therefore materialize only
the two sheared shape columns after a cache reload.

That software invariance check is not evidence that image detection is
physically shear-invariant. An audit of existing ConstGold cases 40--139 found
the full truth response `(1.000008 +/- 0.000063, 0.000118 +/- 0.000065)`, but
the arm-specific detected sample response is
`(0.990795 +/- 0.000295, 0.000183 +/- 0.000300)`. The missing response is
therefore `(-0.009191 +/- 0.000275, +0.000062 +/- 0.000282)`. The deployed
classifier has exactly seven spin-0 inputs and no ellipticity, so it cannot
represent this approximately 0.92% diagonal detection response. It must be
retrained or extended, together with the usable-measurement event described
below, before a precision image-closure claim.

The seven-point importance profile remains supported by
`run_catalogue_closure.py --profile-shears` as the nonlinear MLE fallback. It
requires a supplied shear bracket and is not invoked by the Section 5 runner.
`run_catalogue_closure.py --profile-shears` applies the profiling calculation to
saved Section 5 mocks and records zero-centred finite-difference expansions on
the same fixed importance draws.

For prior atoms `z_j` with catalogue masses `pi_j`, and an optional measured
output cut `W`, it evaluates

```text
A_i(g)   = sum_j pi_j P_det(S_g z_j)
                    p_flow(xhat_i - b_j(g) | S_g z_j)
Ppass_j  = Integral W(xhat + b_j(g)) p_flow(xhat | S_g z_j) dxhat
B_W(g)   = sum_j pi_j P_det(S_g z_j) Ppass_j(g)
p(xhat_i | detected, W, g) = A_i(g) / B_W(g)

b_j(g) = R_blend,j [e(S_g z_j) - e(z_j)]
```

`R_blend,j` is evaluated once at zero shear from the atom's complete
catalogue-neighbour scene and then held fixed.  It is not another latent draw.
The exact finite-shear ellipticity displacement supplies its shear dependence.
Generation adds `b_j(g)` to the two measured-shape outputs; likelihood
evaluation subtracts it before calling the flow; measured-selection QMC draws
add it before applying `W`.  All three operations use the same atom-aligned
`CatalogueModelView.blend_shift` array.

The current flow training table is actually conditional on both cross-matched
detection `D` and a usable four-output measurement `F`; failed/nonfinite ngmix
rows are removed. The deployed classifier predicts `P(D|z)`, not
`P(D and F|z)`. The observed `F` failure rate in the ConstGold pilot is tiny and
cannot explain a percent-level failure, but a final detection/selection claim
must use one canonical usable event `U=D and F`, either by retraining the
classifier for `P(U|z)` or by adding `P(F|D,z)`. `P(U)` then replaces `P_det`
inside both `A_i` and `B`; measurement failure must not be represented as a
continuous `OutputCut` because the flow contains no failure mass.

The measured cut cancels from `A_i` because every retained observation has
`W(xhat_i)=1`, but its population probability does not cancel from `B_W`.
`CatalogueSelection` estimates `Ppass_j` with randomized QMC flow draws whose
latents are held fixed across trial shears, retains only the per-atom pass
fractions, and can persist them in a cut/model/scene-identified cache.  It
evaluates the flow only for positive-mass prior atoms; zero-mass rows remain in
every neighbour-derived model view but cannot contribute their own term to
`B_W`. Persisted arrays have shear-derived stable names and are validated for
row alignment, finiteness, and `[0,1]` support. Array and manifest replacement
is atomic, and the exact cache hashes are recorded in the result. The
closure generator applies the same `OutputCut` after the detection and flow
draws.  `run_catalogue_closure.py` exposes repeatable `--cut-bound
NAME:LO:HI`, `--cut-abs-ehat`, and explicit selection draw/cache settings.

## Reused scene store

`scripts/build_scene_prior.py` turns an explicit user catalogue into two files:

- `galaxies.parquet`, containing the truth rows and optional prior weights;
- `neighbours.npz`, containing a directed CSR neighbour graph and pair offsets.

The graph is built to a user-supplied guard radius and is invariant under trial
shear.  This deliberately matches the image simulations: shear changes only
intrinsic ellipticity.  Positions, pair separations, neighbour identities, and
membership in the 3, 7, or 10 arcsec model apertures remain fixed.

The sheared scene produces distinct model views:

- flow: one primary row plus all-neighbour flux in the configured radial shells;
- detection: one primary row plus only the nearest neighbour within the
  classifier aperture, with an explicit isolated branch;
- response emulator: all supported pairs within its own aperture and `k`, after
  its own training cuts.  Only positive-mass atoms are materialized as
  primaries, while every scene row remains eligible as their secondary.  The
  summed response is cached once for every atom by
  `scripts/build_catalogue_blend_response.py`.

Only intrinsic ellipticity uses the exact finite reduced-shear map.  Magnitude,
circularized half-light radius, positions, separations, and all spin-0 neighbour
summaries are held fixed.  Convergence/magnification is outside this closure
model and a nonzero `kappa` is rejected.

For the existing FS2 image closure, the empirical prior must come from the full
truth scene, not a detected or measured table. The earlier
`realflow_cases0_9_v1` store recovered only 343,341 positive atoms from paired
measurement rows, versus 637,226 eligible atoms in the complete existing raw
cases 0--9, and is therefore not a supported image-closure prior.
`assemble_fs2_scene_catalogue.py` rebuilds the complete 6,995,680-row catalogue
from those already-existing unsheared truth files, verifies their null shear,
and assigns prior mass only inside `18<r<25.8, 0.5<Re<1.5`; it does not render
or simulate new images. Its scene, response, model, and proposal caches have
new provenance hashes and cannot reuse the incomplete store's artifacts.

For the new null prior, preparation is deliberately split into properties and
sky density:

1. `generate_galsbi_prior_base.py` produces independent intrinsic
   `GalSBI("Fischbacher+24")`, model-index-0 catalogues.
2. `build_galsbi_property_bank.py` calls the exact BlendSim quality-processing
   function and merges seeds until the explicit usable-row threshold is met.
3. `randomize_galsbi_scene_cases.py` calls the exact BlendSim realization
   helper for 100 independent cases, each with 71,104 replacement draws and
   uniform positions in one square degree.
4. `build_scene_prior.py` caches the grouped 11-arcsec graph and embeds hashes
   of both upstream manifests.

The supported run uses seeds 20260822, 20260823, and 20260824 and requires at
least one million usable properties. Property-bank size never sets the case
density. Primary mass is nonzero only for `18<r<25.8` and `0.5<Re<1.5`; every
other scene row remains eligible as neighbour context. Scene rows retain
`property_id`, `property_seed`, `case`, and `case_seed` for provenance and
independent-bank tail checks.

## Section 5 null validation

`sbsi.catalogue_null` records the local Newton estimate, sandwich error, score
centring, mean observed information, score variance, their ratio and
delta-method uncertainty, ESS/weight concentration, and score-tail diagnostics
for every component, step, draw rung, and proposal seed. The hard assessment
requires:

- exhaustive and importance estimates on a small reweighted prior to agree
  within 0.25 statistical standard errors;
- the numerical step to agree object by object with exact autograd, and the
  final two draw rungs to agree within 0.25 statistical standard errors;
- score centring and both recovered components to lie within three standard
  errors of zero;
- `E[I]/Var(s)` to lie within 5% of one and within three ratio errors of one;
- score variance and mean information to change by at most 5% across the last
  two draw rungs/proposal seeds, with a Hill tail index above two;
- independently generated, matched `property_seed` prior/mock pairs to be
  statistically compatible, also with tail index above two.

For this runner, the `K=131072` nearest atoms are only a deterministic support
screen. Their actual zero-shear `pi * P_det * L_i(0)` values define the local
proposal before the `M` atoms are drawn. The full proposal remains mixed with
the catalogue prior and every likelihood term retains the exact `pi/q`
correction. This posterior adaptation is a variance reduction only; it does
not change the Section 5 likelihood. The older distance-kernel draw remains
available to the retained profile code.

The validation is staged. `run_section5_powered_exact.py` first draws a small
finite prior, generates its mocks from that same prior, evaluates the exhaustive
sum, and compares numerical derivatives with `autograd_exact_section5`.
It repeats the matched-prior closure for each property bank. Only then does
`run_section5_powered_importance.py` run the 10,000-object full-prior test.
Generating full-prior mocks and merely reweighting the likelihood to a sub-bank
is a misspecified cross-prior test, not an independent-bank closure, and is no
longer a go/no-go criterion. The powered null establishes additive/null closure
only; nonzero-shear tests are still required before a `|m|<0.2%` claim.

The 10,000-object run closes score centring, the 5% information identity, and
the M ladder. Its Hill indices are 2.006 and 1.914, however, so the g2 tail is
still marginal even though its score variance and information are stable to
0.06% across the final two rungs. The powered gate now includes the previously
omitted requirement that the Hill index exceed two; see WORKLOG cont.202.

The supported nonzero follow-up is
`scripts/run_section5_nonzero_closure.py`.  It uses common scene, detection,
flow, and proposal streams at `+-g_inj`, retains aligned object moments, and
estimates the local response with the joint population ratio
`mean[(s+ - s-)/(2 g_inj)] / mean[(I+ + I-)/2]`.  The primary injection is
`g_inj=0.00125`; `0.02` is outside the useful one-step range even though the
nonlinear profile fallback can recover it.  Rotational rings are diagnostic-
only and disabled by default because the current learned likelihood did not
preserve positive mean information on the rotated mock.

At `K=32768,M=8192/16384`, 10,000 ordinary paired scenes give
`m1=-0.3736%+-0.5269%` and `m2=+0.1567%+-0.5044%`, with sub-0.1-sigma rung
movements.  The paired influence Hill indices are 1.759 and 1.742, however,
so the hard tail gate fails and ordinary square-root sample-size projection
is not accepted.  This is a statistically consistent nonzero result, not a
0.2% closure.  Moreover, the off-diagonal responses are
`dg2_hat/dg1=0.1360+-0.0335` and `dg1_hat/dg2=0.1003+-0.0235`.
The five-view diagonal estimator assumes those terms vanish by rotational
symmetry; the runner now rejects them.  A full mixed Hessian needs four more
corner views, while preserving five views requires an actually equivariant
likelihood; see WORKLOG cont.203.

The finite-shear profile diagnostic makes the one-step boundary explicit.
With 10,000 positive-arm mocks and `K=32768,M=16384`, the profile maxima are
`0.020114`, `0.019915`, `0.049809`, and `0.050193` for injected axis shears
`+0.02,+0.02,+0.05,+0.05`, respectively.  Every maximum is within 0.31
profile-curvature errors of truth.  On those same mocks, a Newton step based
only at zero returns about 0.051 and 0.043 for the 0.02 injections, and has
negative zero-point information for both 0.05 injections.  The saved mock
means nevertheless have the same diagonal response at 0.02 and 0.05 to about
four parts in 100,000.  Therefore the failure is not the flow's regularized
mean response: the catalogue mixture has shear-dependent responsibilities and
is not a globally quadratic Gaussian location likelihood.  Finite-shear use
requires profiling or iterative recentering; see WORKLOG cont.204.

For later production use, `select_adaptive_draw_counts` applies per-object
draw doubling on nested zero-shear weight prefixes. It selects the first rung
meeting both an ESS floor and a maximum-weight ceiling, and that chosen prefix
must then remain fixed across the five derivative views. The fixed three-rung
ladder remains mandatory for validation. The null result also reports measured
equivalent flow throughput and projects fixed-`M` GPU-hours per billion
objects, including the per-object candidate pass; adaptive savings must be
benchmarked against that declared baseline.

## Exact closure workflow

Build once:

```bash
python scripts/build_scene_prior.py \
  --catalogue INPUT.feather \
  --output "$DATA_DIR/sbsi/catalogue_prior/scene" \
  --guard-radius-arcsec 11
```

Then run `scripts/run_catalogue_closure.py` from a scheduled job.  All catalogue,
model, observing-condition, aperture, output, and seed values are explicit CLI
arguments.  Supplying `--model-cache` writes the flow/classifier stencil views
on the first run and reloads them on later runs only when the scene files,
model hashes, observing conditions, and apertures match exactly.  The command:

1. draws catalogue scene rows from their stored prior masses;
2. shears every object's intrinsic ellipticity, leaving the rest of the scene fixed;
3. applies a Bernoulli draw from the detection classifier;
4. samples retained measurements from one flow checkpoint and, when enabled,
   adds the fixed external blend shift;
5. scores those measurements against every catalogue atom at a common
   finite-difference stencil;
6. saves measurements, generating scene rows, detection uniforms, seeds, model
   and implementation hashes, the directional Newton estimate, and both
   sandwich and model-based standard errors.  Exact runs also retain the
   per-object log likelihood, score, and information so finite-difference and
   seed checks can be paired rather than inferred from aggregate totals.

One flow checkpoint is intentional for closure: generation and likelihood must
be the same density.  An ensemble is a later robustness experiment, not the
first exactness test.

This is a likelihood closure, not automatically an image-simulator closure.
Every loaded image mock must carry a current manifest whose mock kind,
intrinsic-ellipticity-only shear transform, measurement-model hash, target
order, injected shear, and table hashes agree with the loaded data. A loaded
likelihood-generated mock is stricter: it must carry the numerical runner's
full scene/model/response/selection generation identity, implementation
hashes, and output hashes. Paired summaries re-hash both analysis tables,
check row counts, convergence, positive final information, importance
diagnostics, full run identity, and non-overlapping simulation-case blocks.
Their run identity includes proposal method, reference shear, initial-value
reuse, and the candidate/reuse/numerator/selection/diagnostic flow-evaluation
split, so results from different proposal constructions cannot be combined.
Its prior population may be changed without invalidating the algebra because
the same finite prior and measurement likelihood generate and score the mock.
But an image closure additionally requires that the flow, detection
classifier, and response emulator are valid for the simulated population.
Model and catalogue provenance must therefore label these two claims
separately; a likelihood closure on a new population does not demonstrate
population transfer of models trained elsewhere.

Image mocks are a distinct input type.  The pixels already contain blending,
so they must not carry a catalogue-prior `scene_row` or receive the analytic
`R_blend` injection used by likelihood-generated mocks.  The inference model
still includes its fixed catalogue-atom response when evaluating the prior.
`scripts/prepare_image_closure_mock.py` reads a single declared BlendEMU image
leg through its input-truth, cross-match, and shape catalogues; verifies that
every rendered input has the requested coherent `(g1,g2)`; keeps each detected,
source-selected object once; rejects failed ngmix fits; and maps `NGMIX_G1`,
`NGMIX_G2`, `MAG_AUTO`, and `log(FLUX_RADIUS)` to the checkpoint's four output
names.  It hashes every source catalogue and tags the frozen mock as `image`.
The closure runner validates this tag and source identity and deliberately
skips only the likelihood-mock `scene_row`/injected-shift check. If a measured
output cut is requested, the numerical runner applies that exact predicate
independently to each loaded image leg, records its keep fraction, and freezes
the retained rows in its output mock before proposing atoms.

`measured_log_flux_radius` is the natural logarithm of SExtractor
`FLUX_RADIUS` in pixels. At 0.2 arcsec/pixel, a measured radius above 0.5 arcsec
means `measured_log_flux_radius >= log(2.5) = 0.916290732`; it is not
`log(0.5)`. The existing images put almost every valid fit above this threshold,
so `logR>=1.45` is retained as a stronger selection stress test.

The image adapter already conditions on a catalogue cross-match and rejects
failed ngmix fits. Before interpreting the image result as a calibrated
detection closure, verify that this measurement-success event is represented
by the detector/conditional-flow definition; the adapter reports the dropped
count rather than silently treating it as modeled.

Use one image leg rather than an antithetic response catalogue for this test.
An antithetic catalogue generally conditions on detection and successful shape
measurement in both legs, which is not the selection of a real single-image
sample.  The two legs can instead be prepared and inferred independently, with
case blocking used later when estimating shear bias.

For an external-response run, first build the cache once. The general closure
runner takes both `--inject-r-blend` and `--blend-response-cache`; the numerical
recenter runner enables it by supplying `--blend-response-cache`. Both verify
hashes of the scene store, emulator artifacts, response cache, and observing conditions.
Response-aware model, proposal, and selection caches have a distinct identity.
Frozen mocks are also rejected unless their recorded per-object response and
two-component shift agree with the supplied cache.

## Defensive importance sampling

`sbsi.catalogue_sampling` now supplies the first scalable candidate.  For each
measured galaxy `i`, the proposal is

```text
q_i(j) = epsilon pi_j + (1-epsilon) q_local,i(j).
```

The base local component is a distance kernel over the `K` nearest catalogue
atoms in a low-dimensional measured-property space.  The coordinates of each atom
are computed once from QMC draws of the complete flow and cached; the explicit
mean head is not used because it need not equal the full conditional mean.
The neighbour tree contains only positive-mass prior atoms.  Zero-mass scene
rows remain available when constructing each atom's clustered environment but
cannot consume proposal candidates or draws.
The local base mass is `pi_j P_det(z_j)` at zero shear.  The defensive component
draws from the complete `pi_j`, so every positive-prior atom remains supported
and `pi_j/q_i(j) <= 1/epsilon`.

The default Section 5 null runner adapts that candidate support with the exact
zero-shear likelihood. Numerical recentering now performs the same exact
adaptation at its declared initial centre, then freezes the resulting draw for
the complete optimization. The local adapted mass on the `K` candidates is
proportional to `pi Pdet L(initial)`; mixing with `epsilon pi` preserves full
support and the exact importance identity. Candidate evaluations reused at the
initial point are accounted separately. The bandwidth field is inactive for
this adapted method. Other retained catalogue-profile interfaces and the
explicit `distance_kernel` recenter fallback continue to use the base distance
kernel.

Only the expensive numerator is sampled:

```text
Ahat_i(g) = (1/M) sum_m [pi_jm/q_i(jm)] P_det(S_g z_jm)
                                      p_flow(xhat_i | S_g z_jm).
```

`B(g)` remains an exact sum because classifier predictions are already cached
for every atom.  Every per-galaxy draw matrix is fixed across the full shear
stencil, and draw ladders are nested prefixes.  Each object owns a
deterministic RNG stream and each draw consumes one fixed-width random record,
so requesting a larger maximum M cannot shift later objects or any other draw
field.  A regression checks exact prefix equality for indices, probabilities,
mixture/membership flags, and candidate radii.
Full-versus-object-chunked adapted draws are also identical in atom IDs,
proposal probabilities, local/global flags, candidate radii, and local
positions. A full-model regression checks that reused initial candidate
likelihoods reproduce an independently evaluated fixed-draw surface with both
selection normalization and `R_blend` enabled.

`scripts/run_catalogue_closure.py --sampler both` runs the exact oracle and IS
on the same mock catalogue.  Importance arguments are explicit.  In
particular, `--proposal-candidates` accepts several `K` values and
`--proposal-seeds` accepts several independent seeds.  Optional `--gate-*`
thresholds produce one stopping decision only when all six thresholds are
supplied together.  The gates require:

- exact-oracle shear agreement;
- stability between the last two `M` rungs;
- agreement across independent proposal seeds;
- agreement as `K` expands;
- a minimum mean ESS fraction;
- a maximum 90th-percentile single-weight fraction.

ESS cannot override failure of any other gate.  Results also record the
importance contribution from atoms outside the local candidate set and from
draws generated by the defensive global component.

On the real one-case oracle, magnitude-only and magnitude+size proposals were
insufficient.  A proposal using all four measured flow outputs passed every
predeclared gate at `M=4096,16384` and `K=8192,16384`.  These values scale with
the support size rather than being universal: enlarging the empirical prior
requires a corresponding K expansion, checked through the reported
outside-local contribution.

## Validation gates and next phase

Before replacing the exact sum, the real-model small-catalogue run must pass:

- centred score at zero shear and recovery of small positive and negative
  injections, assessed using the object-level sandwich uncertainty rather than
  a fixed residual threshold;
- finite-difference convergence under changes to `delta` and Richardson use;
- stable results across disjoint catalogue shards and random streams;
- cache feature parity with the flow and classifier checkpoint metadata;
- agreement between cached response-pair preparation and BlendEMU.

The current flow features contain hard 3- and 7-arcsec neighbour shells, but
their membership is fixed under the project's shape-only shear convention.
They therefore cannot create shear-dependent boundary crossings.  A finite
empirical prior can still give a granular curvature through its discrete shape
support and nonlinear flow/detection response.  The powered closure must use
enough independent scene cases to stabilize both the score response and the
score-squared/observed-information identity; a good IS-versus-exact comparison
on one small case does not by itself establish this.

The scalable path therefore distinguishes the score diagnostic from the final
estimate.  `score_and_information(..., center=(g1,g2))` can iterate the score
about a nonzero trial shear and reports both observed-curvature and
score-squared (Fisher) steps.  For a more robust final MLE,
`--profile-shears` evaluates the likelihood itself on a directional shear grid
using identical importance draws at every point.  Nested M prefixes and
independent proposal seeds expose Monte Carlo movement of the maximum.  A
three-point quadratic interpolation is accepted only for an interior,
concave local maximum; otherwise the reported estimate remains the grid
maximum.  The implementation holds all requested proposal draws but only one
multi-million-row model view at a time, rebuilding it with the attached
detection table and releasing it before the next shear. Under the declared
spin-0 checkpoint, nonzero views reuse zero-shear conditions and detection,
change the two primary shape columns, and apply the atom-aligned blend target
shift. This bounded
memory scan is the operational form of iterating the documented MLE rather
than trusting one Newton step from zero.

Closure reruns should pass `--mock-input OUTPUT_OF_GENERATION` rather than
regenerating measurements.  GPU flow sampling is statistically reproducible
but not bitwise identical across device types; changes at the `1e-6` level can
alter a target-specific nearest-neighbour proposal.  The frozen mock files are
hashed and validated before use, so nested-M, proposal-seed, and grid-spacing
comparisons vary only the quantity named by the check.

The ten-case N=2,048 closure with R_blend disabled passes at M=32,768/65,536
and K=131,072.  The two proposal seeds recover odd shear amplitudes 0.020282
and 0.020155 for injections of +/-0.02, with likelihood errors 0.001171 and
0.001376.  The maximum M movement is 0.001198, seed span 0.000929, and
0.005-to-0.0025 profile-spacing movement 0.000494.  Zero, each sign, and the
even offset all lie within the predeclared three-sigma gates.  These numbers
establish initial flow+detection catalogue-prior closure; they do not yet
validate a nonzero R_blend contribution.

Measured-output selection is now closed on the same ten-case prior.  The
likelihood divides each retained object by
`sum_j pi_j Pdet_j Ppass_j`, with `Ppass` evaluated from common-random-number
draws of the four-output flow.  N=2,048 zero/sign profiles pass for the isotropic
shape cut, the combined shape-plus-size cut, and the strong size-only cut
`measured_log_flux_radius >= 1.45`.  For the size-only panel the two proposal
seeds give `(-0.02210,-0.02248)`, `(-0.00017,+0.00056)`, and
`(+0.01848,+0.01838)` at injected shears `(-0.02,0,+0.02)`; a conservative
three-mock fit gives `m=+1.33% +/- 6.16%` and
`c=-0.00121 +/- 0.00101`.  This establishes the normalization and strong-cut
closure at the available precision, not a sub-percent bias limit; that requires
the independent multi-mock calibration.

That calibration now uses five paired random-stream blocks at each of
`g=(-0.02,0,+0.02)`, independently for both components.  For the same
size-only selection it gives
`g1: m=-1.7135% +/- 1.5296%, c=+0.000396 +/- 0.000252` and
`g2: m=+0.9336% +/- 1.9543%, c=+0.000315 +/- 0.000389`.
All 30 profiles have interior concave maxima; the independent audit reproduces
the summaries exactly, and every arm is within two standard deviations.  This
closes the current flow+detection+measured-selection implementation at roughly
two-percent multiplicative precision.  It is consistency with zero, not yet a
sub-percent bound.

The defensive importance implementation has passed the available real-flow
oracle and multi-case profile checks under draw-count, candidate-radius,
tail-contribution, and independent-seed expansions.  Tempered SMC remains a
per-object fallback if later, broader priors defeat those convergence checks.
The fixed real-emulator response now passes its frozen-M convergence test:
shared M=32,768 likelihoods reproduce bit-for-bit across runs, M=65,536
movement is below 0.000082, and proposal-seed span is 0.000701.

Those likelihood-generated tests do not yet close the existing ConstGold
image path. The current image ordering is strict: first stabilize the uncut,
full-`R_blend` fit across proposal seeds and draw rungs; next reconcile the
approximately 0.92% detection response and the separate usable-measurement
event `U=D and F`; only then enable the realistic or stress measured-output
cuts. Existing selected likelihood closures validate the likelihood formula
and normalization, not image selection. Until the uncut sampler gate passes,
image runs retain `selection=null` and perform zero selection-flow evaluations.

A separately generated GalSBI/Fischbacher+24 base and 50 randomized-position
cases supply a 315,799-atom disjoint likelihood-closure prior.  Five paired
mock blocks at each of `g=(-0.02,0,+0.02)` give
`g1: m=-0.3564% +/- 0.3393%, c=-0.000082 +/- 0.000682` and
`g2: m=-0.8042% +/- 0.9441%, c=-0.000516 +/- 0.000660` with all 30 profile
maxima interior, concave, and inside 2.28 sigma.  This is the powered
flow+detection+fixed-Rblend likelihood closure.  All deployed models were
trained on FS2-25876, however, so a GalSBI image closure would be a
population-transfer test and must not be reported as matched-population
validation.

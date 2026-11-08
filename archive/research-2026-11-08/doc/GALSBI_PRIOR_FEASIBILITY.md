# GalSBI as an evaluable population prior

## Scope and verdict — September 30, 2026

The owner asked whether GalSBI can evaluate the posterior density of an
arbitrary true galaxy, using uniform positions for neighbour environments
and excluding the clustering extension. The initial source/mathematical
audit below is followed by the subsequently authorized
[isolated-galaxy implementation](#isolated-galaxy-implementation--september-30-2026)
and its catalogue comparison, followed by the
[uniform-neighbour extension](#uniform-neighbour-implementation--september-30-2026).
These stages do not test shear-inference closure.

Subsequent work implements an observation-indexed nine-summary likelihood and
small standard-MH prototypes; see
[sampling implementation and evidence](CONTINUOUS_SAMPLING.md#fixed-summary-prototype--september-30-2026).
The nine-summary joint law is sampleable but has no general smooth-density
evaluator. The native-to-SBSI radius mapping discussed as unresolved below
was subsequently established from the installed UFig rendering matrix in
that implementation. No sub-percent shear validation follows from this.

The non-clustering **Fischbacher+24 phenomenological model** has explicit
population distributions from which an evaluable joint latent prior can be
constructed. The inspected package does **not** expose a complete galaxy
`log_prob` API. Its posterior loader supplies samples of population-model
parameters, not a density evaluator for individual galaxies. A marginal
density in SBSI's five intrinsic coordinates requires additional integration.
An augmented latent-variable sampler can avoid that inner marginalization.

This changes the earlier proposal to fit a Gaussian mixture: first consider
using GalSBI's existing physical/statistical factors directly. It does not
establish that GalSBI describes the FS2 parent of current ConstGold tests.

## Evidence and source identity

Inspected the public repository at commit
`df802e1db2f2a4247955720aea227d5b65cdf975` using a shallow read-only checkout at
`/tmp/sbsi-galsbi-audit-20260930`. No upstream module was imported or executed,
no GalSBI dependencies or data bundles were installed, and no GPU job ran.
Although this revision includes Fischbacher+26, the assessed configuration
and model factors are those selected by Fischbacher+24 with uniform positions.
The online documentation has mixed version labels, so code claims below are
pinned to the inspected commit rather than inferred from a version label.

Primary references:

- [Phenomenological science paper](https://arxiv.org/html/2412.08701v3),
  sections 2.1 and 2.4: galaxy model and ABC parameter inference.
- [Population-model documentation](https://cosmo-docs.phys.ethz.ch/galsbi/galpop_pheno.html):
  luminosity, SED, size, ellipticity and light-profile distributions.
- [Catalogue interface](https://cosmo-docs.phys.ethz.ch/galsbi/getting_started.html):
  `model_index` chooses a posterior parameter sample.
- [Pinned source tree](https://gitlab.com/cosmology-ethz/galsbi/-/tree/df802e1db2f2a4247955720aea227d5b65cdf975/src/galsbi).

Source paths below are relative to that upstream `src/galsbi` tree.

| Source | Verified behavior |
| --- | --- |
| `galsbi.py` | Public methods are `generate_catalog`, `cite`, `load_catalogs`, and `load_images`; generation loads posterior rows and passes one parameter row to the workflow. |
| `load.py:20` | `load_abc_posterior` returns structured HDF columns from the posterior file. |
| `configs/config_Fischbacher+24_intrinsic.py` | Selects truncated-logexp luminosity evolution, Dirichlet-alpha-mode SED coefficients, SDSS-fit sizes, beta-mode-red-blue shapes, and beta-prime-red-blue Sérsic indices. |
| `ucat/galaxy_population_models/galaxy_luminosity_function.py:541` | `LuminosityFunction.analytical_number_density(m,z)` evaluates a Schechter number density in Mpc^-3 mag^-1; this is not a normalized full-galaxy density. |
| `ucat/galaxy_population_models/galaxy_sed.py` | Converts mode/scatter parameters to redshift-dependent Dirichlet concentrations; the F24 template weights are all one. |
| `ucat/galaxy_population_models/galaxy_size.py:137` | Draws a normal variate, applies the magnitude-dependent mean, exponentiates with `np.exp`, and multiplies by a redshift factor. |
| `ucat/galaxy_population_models/galaxy_shape.py:106` | Draws a scaled beta ellipticity amplitude and uniform complex phase. |
| `ucat/galaxy_population_models/galaxy_light_profile.py:79` | Rejection-samples beta-prime Sérsic indices inside configured bounds. |
| `ucat/plugins/sample_galaxies_photo.py:510` | Converts SED coefficients and luminosity-function magnitude into apparent magnitudes, including filter, distance and extinction calculations. |
| `ucat/plugins/sample_galaxies_morph.py` | Conditions size on the absolute magnitude in the configured reference band, not automatically the luminosity-function band. |

Validation performed: source search for density/probability/likelihood APIs,
AST inspection of the interface and eight supporting modules, and manual
tracing of generation and transformations. No complete prior density was
numerically normalized or compared against generated catalogues in this audit.

## Three distinct densities

Let `Theta` denote global population-model parameters and `D` the observations
used by GalSBI's ABC analysis.

1. `p(Theta | D)` is the posterior over population parameters. The package
   loads its sample representation; it does not thereby provide an arbitrary
   point evaluator for this posterior.
2. `p(u | Theta)` is the distribution of an individual galaxy's native latent
   variables at fixed population parameters. Its factors are explicit.
3. `p(x | D) = integral p(x | Theta) p(Theta | D) dTheta` is a galaxy's
   posterior-predictive population density, where
   `x = (e1, e2, true_r_magnitude, circularized_Re, sersic_n)` for SBSI.

The third is the proposed shear-inference prior. It is not an individual
galaxy's posterior conditioned on that galaxy's measured data.

Posterior samples can approximate the integral over `Theta` without a
continuous evaluator for `p(Theta | D)`. Use the supplied posterior weighting
convention; it has not been audited by downloading the posterior bundle here.
For a first computational closure test, freeze a predeclared posterior row.
For population-uncertainty propagation, `Theta` is shared by the target,
neighbours and ultimately the catalogue. Independently averaging a different
`Theta` for each galaxy is not equivalent to marginalizing one common
population model over the full catalogue likelihood.

## Evaluable joint prior in native latent coordinates

Use galaxy type `t` (blue/red), redshift `z`, absolute B magnitude `M_B`,
five SED fractions `c` with four independent coordinates, physical half-light
radius `R`, Sérsic index `n`, ellipticity amplitude `a`, and complex phase
`psi`. At fixed foreground/observing conditions, a schematic intensity is

```text
lambda_t(u | Theta)
  = phi_t(M_B,z | Theta) * dV/(dz dOmega)
    * Dirichlet(c | alpha_t(z,Theta))
    * LogNormal(R | mu_t(M_ref(M_B,c),z,Theta), sigma_t(...))
    * TruncatedBetaPrime(n | z,t,Theta)
    * ScaledBeta(a | t,Theta) / (2*pi)
    * 1_domain(u).
```

This expression is a derivation from the source, not an existing GalSBI API.
It is a density in the stated native coordinates (and counting measure for
type). Sum and integrate the intensity over the declared finite population
domain to obtain surface density `Sigma(Theta)`. Dividing by `Sigma` produces
a normalized random-galaxy prior. Red/blue probabilities then follow from
their integrated intensities; they are not arbitrary equal mixture weights.

Details essential to an implementation:

- The luminosity function needs the comoving-volume factor for a sky
  population. Its physical density alone is insufficient.
- Sérsic rejection bounds require the beta-prime CDF normalization at every
  redshift/type/parameter setting. The current F24 configuration inherits
  `sersic_n_min=0.2` and `sersic_n_max=10`; the sampling function's standalone
  default maximum is different. Read the resolved configuration.
- The beta distribution is for **radial amplitude**, not Cartesian shape.
  In the same shape convention, density in `(e1,e2)` is
  `p_a(a)/(2*pi*a)` for `a>0`. Sampling amplitude and phase avoids having to
  evaluate the Cartesian singularity at zero. Further shape-convention
  transformations have their own Jacobians.
- Size sampling uses natural-log scatter even though the mean expression
  contains a base-10 logarithm. Angular conversion and circularization must
  follow explicit conventions.
- SED fractions are coupled by their simplex constraint. Apparent r magnitude
  is a derived quantity; neither independent coefficients nor a standalone
  apparent-magnitude Schechter density reproduces this model.
- `M_ref` for the size relation is the reference-band absolute magnitude
  generated from the spectrum. F24 uses i as reference and B for luminosity.

## Why the five-dimensional density is additional work

The available SBSI truth vector does not identify redshift, type or SED.
To evaluate its density, those variables must be marginalized. At fixed
`z,c,t`, the luminosity amplitude can be chosen to reproduce the supplied
true r magnitude and the physical size chosen to reproduce its angular size.
The result can be expressed schematically as

```text
p(x | Theta) = sum_t integral lambda_t(u(x,z,c,t) | Theta)
                            * J(x,z,c,t) dz d^4c / Sigma(Theta).
```

`J` contains the appropriate coordinate Jacobians and the integrand includes
the population-domain indicator. There is no verified built-in implementation
of this marginal in the inspected package. It is a five-dimensional nuisance
integral even before averaging over population parameters.

For inference, a simpler first route is to **keep those latent variables in
the sampler**. Evaluate the explicit native joint prior, map each latent draw
to the five SBSI physical features, rebuild the neighbour-dependent features,
and evaluate the existing C8/F8/disk likelihood. Marginalization then happens
through sampling instead of a new inner quadrature at every proposed galaxy.
This increases sampling dimension; it is not a demonstrated speed improvement.

## Continuous model versus the released catalogue sampler

The underlying analytic model is continuous, but its numerical catalogue
implementation is not entirely continuous:

- `RedshiftAbsMagSampler.__call__` draws redshift on `z_grid` and uses a
  nearest inverse-gamma-table index for absolute magnitude.
- `MagCalculatorTable.__call__` uses nearest grid indices for redshift and
  foreground extinction when evaluating precomputed spectral integrals.
- The default catalogue output precision is float32.

A density using continuous analytic factors therefore targets the continuous
model, not exactly every discretization in an unmodified generated catalogue.
Before attributing a closure difference to inference, refine/check these
grids or use consistent continuous transformations. Direct spectral evaluation
is available, but its accuracy and runtime have not been tested here.

## Uniform neighbour environments

Under the owner's unclustered approximation, use a marked homogeneous Poisson
process at fixed `Theta` and fixed foreground conditions. In an aperture of
area `A`, an ordered iid neighbour-list representation has

```text
N | Theta ~ Poisson(Sigma(Theta)*A)
positions_j | N ~ Uniform(area A)
u_j | Theta ~ p_gal(u | Theta)

p(u_target,N,{position_j,u_j} | Theta)
 = p_gal(u_target | Theta) * Poisson(N | Sigma*A)
   * product_j [p_gal(u_j | Theta)/A].
```

For a circular aperture of radius `Rmax`, draw separation as
`Rmax*sqrt(U)` and angle uniformly, not separation uniformly. Neighbour counts,
faint population boundary and aperture coverage must all be explicit. Uniform
positions alone do not specify the density or flux distribution of neighbours.
Include neighbours below detection cuts and construct the three close-crowding
summaries and disk pairs from the same neighbour realization. Use the maximum
aperture required by the retained models, including the 10-arcsec F8/C8 guard.

This finite-aperture model has an explicit joint density. MCMC with varying
neighbour count still needs correctly weighted count-changing moves or an
equivalent integration method; uniform positions do not eliminate that work.
A computational first test can condition on a fixed count, but must be labelled
conditional rather than presented as complete population inference.

## Compatibility and a bounded next step

The F24 default intrinsic catalogue is restricted to apparent reference-band
i magnitude 14–28, with further redshift/absolute-magnitude generation bounds.
These are population support choices, distinct from our measured r<25.8 and
radius>0.6 arcsec selection. Blindly taking that default could omit latent
galaxies that scatter into our sample or contribute neighbour flux. The
surface-density normalization must use whichever population domain is adopted.

The inspected GalSBI lensing path defaults to distortion ellipticity, whereas
SBSI's scene shape convention uses `(1-q)/(1+q)`. Establish the conversion
before evaluating F8/C8. Also verify whether GalSBI's half-light radius maps
directly to SBSI's circularized radius; the catalogue column descriptions alone
do not settle that convention. Pin filter throughput, units and foreground
extinction; an r-band label alone is not an adequate mapping contract.

ConstGold was generated from FS2, not this fitted GalSBI population. Replacing
its inference prior is a population-change experiment, not a pure improvement
to the original integral. A GalSBI-generated model-closure test can assess a
new inference implementation but cannot prove recovery on the FS2 population
or validate the retained models outside their supported feature domain.

Recommended first implementation: one F24 posterior row, one explicit
population domain, analytic log-density factors in native coordinates,
deterministic mapping to SBSI, and the unclustered neighbour construction.
Validate each factor and the composed joint distribution against the matching
generator, including density normalization, type weights, coordinate
Jacobians and grid convergence. Then run a small observation-level posterior
and shear-profile test with recomputed measured-selection normalization.
Keep posterior-row uncertainty and full-scale shear validation as later stages.

The source audit above did not produce an adapter or numerical validation.
The subsequently authorized isolated-galaxy implementation is described below.

## Isolated-galaxy implementation — September 30, 2026

`sbsi/galsbi_prior.py` implements the native joint intensity above, including
the volume element, Dirichlet normalization, size Jacobian and conditional
Sérsic truncation. Its input measure is counting measure in type and
Lebesgue measure in `z, M_B, c[0:4], R_kpc, n, a, psi`, with the fifth SED
fraction constrained by the other four. `a` is native GalSBI distortion;
`psi` is the complex-shape phase in radians. This is a latent-variable prior
at one fixed population-parameter row, not a marginal five-feature density
or an average over the ABC population posterior.

The class provides `log_intensity(data)`, `log_prob(data, surface_density=...)`,
and independent continuous `sample(n_proposals, seed)` draws. The last method
returns a randomly sized retained catalogue and a Monte Carlo estimate of
the surface-density normalizer with its standard error. The proposal count
is fixed before applying the intrinsic reference-i limits; it does not keep
drawing until a requested retained count is reached. For fixed population
parameters the normalizer cancels from within-population MCMC ratios.
Changing population parameters requires their corresponding normalizers.

`scripts/check_galsbi_continuous_prior.py` supplies GalSBI photometry and
cosmology callbacks, generates a separate GalSBI reference catalogue, and
compares it with the analytic implementation. It uses installed GalSBI 0.3.0,
whose source differs from the upstream checkout in the initial audit.
Actual installed source files, posterior/table data and reference catalogues
are hashed in `result.json`. GalSBI remains an optional diagnostic dependency;
the core density module imports only NumPy and SciPy.

The pilot freezes accepted F24 posterior row 0 (source id 8), HSC r2/i2
filters, zero foreground extinction and intrinsic reference-i magnitude
14–28. It retains the generator's redshift/absolute-magnitude computational
envelope, including `M_B <= -4`; GalSBI itself warns this faint boundary can
exclude some intrinsically faint galaxies. Matching this support tests
implementation consistency, not completeness for SBSI measured selection.
No neighbours or measured selection enter this comparison.

Run from the editable-installed SBSI checkout via the CPU scheduler:

```bash
export GALSBI_CHECK_OUTPUT=/absolute/path/to/new/run_directory
mkdir -p "$GALSBI_CHECK_OUTPUT"
sbatch --output="$GALSBI_CHECK_OUTPUT/comparison_%j.out" jobs/job_galsbi_prior_check.sh
```

The default stage is `all`;
`GALSBI_CHECK_STAGE=compare` reuses the saved reference catalogue, requiring
matching generation metadata. The script also exposes `--model-index`,
`--nside`, `--reference-seed`, `--sample-seed` and `--proposals` for explicit
CPU runs. Default settings are recorded in the generated manifests.

Density evaluations for every reference and continuous galaxy are retained
in `density_evaluations.npz`; `continuous_catalogue.npz` stores the continuous
draws and derived photometry. `parameters.json` records resolved parameters.
`population_comparison.png`/`.pdf` and `conditional_density_checks.png` show
distribution and conditional-CDF comparisons. A failed diagnostic writes its
evidence and exits unsuccessfully; passing checks do not establish equivalence
or sub-percent accuracy.

The remaining inference work includes the native-to-SBSI shape/radius mapping,
neighbour environments, measured-selection normalization and sampler closure.
Photometry still shares GalSBI's nearest-grid spectral tables, so this is an
evaluable continuous latent law, not a globally smooth differentiable model.

## Isolated-galaxy pilot results — September 30, 2026

Artifacts: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_continuous_prior_20260930_v1`.
Reference seed 2026093011 and continuous seed 2026093012 were fixed before
comparison. One HEALPix pixel at nside 512 defines the reference area.
The comparison completed as CPU job 16775868 in 40 seconds, using about
0.80 GB peak reported RSS and no GPU. The full test suite completed in job
16775847: **328 passed, 3 skipped**, in 43.31 seconds. Ruff and shell syntax
checks passed. Final source snapshots and their comparison hashes were
verified; plots were visually inspected.

| Check | Result |
| --- | --- |
| Actual GalSBI reference | 11,669 galaxies: 520 red, 11,149 blue |
| Independent continuous draws | 70,507 retained from 131,072 fixed proposals; 3,067 red, 67,440 blue |
| Pointwise r, i, absolute-i magnitude replay | Maximum absolute error 1.42e-14 mag |
| Native angular radius replay | Maximum absolute error 8.88e-16 arcsec |
| Native log intensity | Finite for every reference and retained continuous galaxy |
| Population normalization | (2.91704 +/- 0.00747)e9 sr^-1; Monte Carlo relative SE 0.256% |
| Reference-area count prediction | 11,652.83; observed-minus-expected 0.144 standard deviations using Poisson plus normalization uncertainty |
| 15 one-dimensional distribution checks | KS maximum CDF discrepancies 0.00549–0.01284; smallest raw p=0.07295 |
| Five coarse two-dimensional checks | Smallest raw p=0.13098 |
| Red/blue fraction check | Raw p=0.60256 |
| Morphology conditional CDF checks | Raw p: size 0.4298, Sérsic 0.9547, shape 0.04835, phase 0.7878; minimum Holm-corrected p=0.1934 |
| Shape/Sérsic density integrals | Maximum absolute error from unity 1.09e-12 |
| Coarsened luminosity integration grid | Maximum relative rate change 1.81e-9 |
| Conditional magnitude inverse CDF | Maximum probability error 2.35e-8 |

All nine declared pilot gates passed. The 21 catalogue-distribution checks
form one Holm-corrected family; the four conditional-CDF checks form another.
Both families use a 0.01 rejection threshold. The weak raw shape-CDF p-value
is retained above rather than hidden by the correction. These are discrepancy
detection tests, not equivalence tests or evidence that errors are below 0.3%.
The normalization's 0.256% Monte Carlo uncertainty is a population-count
quantity and says nothing about shear bias. Fine joint structure, rare bright
tails, other population-posterior rows and downstream inference remain
unvalidated.

## Uniform-neighbour implementation — September 30, 2026

`sbsi/galsbi_scene_prior.py` adds `GalSBIScenePrior` and `GalSBISceneBatch`.
The latter holds native target dictionaries, native neighbour dictionaries,
CSR `indptr`, and `offset_arcsec` with two Cartesian coordinates per
neighbour. Each target has its own independent patch; this is not a spatially
shared catalogue for multiple nearby observations. The default radius is
10 arcsec, matching the F8/C8 crowding guard. Counts are not capped and may
be zero. Targets and neighbours share one fixed population-parameter row
and the same intrinsic population domain, including galaxies fainter than
the SBSI measured magnitude cut.

The generator draws candidate neighbour counts from
`Poisson(parent_surface_density * aperture_area)`, assigns candidates to
patches, and applies the galaxy prior's ordinary intrinsic population cuts.
Poisson thinning yields the intended retained count and mark law without
using an estimated retained surface density to draw counts. Retained
`proposal_index` values preserve patch membership. Positions use
`radius = aperture * sqrt(U)` and a uniform angle. Fixed target counts use
ordinary rejection sampling separately; those stopped draws do not estimate
population normalization.

For `C = (180/pi*3600)^2` arcsec² per sr, `A` in sr, retained intensity
`lambda(u)` per sr, and retained surface density `Sigma` per sr, the ordered
neighbour-list density is

```text
log p(scene) = log lambda(target) - log Sigma
             - Sigma*A - log(N!)
             + sum_j [log lambda(neighbour_j) - log C].
```

This density is in the native galaxy coordinates, counting measure on `N`,
and Cartesian arcsecond offsets within the disk. There is no extra radial
Jacobian in these Cartesian coordinates. `log_kernel(scene)` retains the
count-dependent factorial and omits only `-Sigma*A-log Sigma`, constant for
fixed population parameters and aperture. `log_prob(scene, surface_density=...)`
includes those terms, and is only approximately normalized if `Sigma` is a
Monte Carlo estimate. Do not drop the factorial when changing neighbour
counts. Do not sort the latent list by radius and then claim its density is
with respect to iid ordered coordinates. No count-changing MCMC kernel is
implemented here.

`v39_crowding(scene)` computes the three F8/C8 values directly from neighbour
magnitudes and physical offsets. The full marked scene remains available;
the prior is not a fitted density over just three crowding summaries. For an
aperture larger than10 arcsec, this projection still uses only the inner10;
an aperture smaller than10 is rejected for this projection. Recompute native
photometry after changing redshift, luminosity or SED before evaluating these
features. Native radius/shape conversion and the emulator remain future work.

Run the bounded CPU diagnostic using the completed isolated comparison:

```bash
export GALSBI_CHECK_REFERENCE=/absolute/path/to/isolated_comparison
export GALSBI_CHECK_OUTPUT=/absolute/path/to/new/neighbour_comparison
mkdir -p "$GALSBI_CHECK_OUTPUT"
sbatch --output="$GALSBI_CHECK_OUTPUT/comparison_%j.out" jobs/job_galsbi_neighbours.sh
```

The reference diagnostic uses the actual GalSBI-generated marks and catalogue
count, assigns each galaxy once to fresh uniform square-cell positions, then
keeps disjoint circular apertures inside the cells. Cell-corner exclusions are
reported. This explicitly tests the requested uniform-position model rather
than original HEALPix geometry or clustering. The two arms share target marks
for paired crowding comparisons; their neighbour populations are independent.
Saved `.npz` files contain full native scenes, crowding and density evaluations.

Within a scheduled Python process, the minimal API is:

```python
from scripts.check_galsbi_continuous_prior import build_prior
from sbsi.galsbi_scene_prior import GalSBIScenePrior

galaxy, _, _ = build_prior(model_index=0)
prior = GalSBIScenePrior(galaxy, aperture_arcsec=10.)
scenes, metadata = prior.sample(4096, seed=2026093021)
log_prior_kernel = prior.log_kernel(scenes)
close_crowding = prior.v39_crowding(scenes)
```

The neighbour dictionary contains every native galaxy variable and its true
r/i photometry; offsets and CSR membership identify its target. It is enough
to retain the latent scene while working on the remaining physical-coordinate
mapping. The Gaussian summaries themselves are derived features, not additional
independent prior variables.

## Uniform-neighbour pilot results — September 30, 2026

Artifacts: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_neighbours_20260930_v1`.
Fixed seeds are2026093021 for continuous scenes and2026093022 for the reference
placement. The population, filters and intrinsic cuts are unchanged from the
isolated pilot. No measurement-selected parent, C8 thinning, flow outputs or
emulator evaluation enters this experiment.

| Check | Result |
| --- | --- |
| Continuous scenes | 4,096 targets, 87,622 neighbours retained from163,663 Poisson envelope candidates |
| Mean count within10 arcsec | 21.392 versus21.540 expected; discrepancy -1.62 standard deviations including population-normalization uncertainty |
| Count variance / mean | 0.99429; Poisson expectation1 |
| GalSBI reference patches | 424 disjoint circles, 9,134 neighbours; 2,535 cell-corner rows excluded, no row reuse |
| Reference mean count | 21.542; discrepancy0.011 standard deviations |
| Spatial uniformity | Smallest raw p0.03464 (continuous azimuth); minimum Holm-corrected p0.13854 acrossfour tests |
| Neighbour galaxy distributions | All21 comparisons pass; smallest raw p0.08022, all Holm-corrected p1 |
| Crowd feature means, continuous minus reference | At0.5,1,2 arcsec: -0.00175 +/-0.03555, -0.03996 +/-0.05649, -0.03800 +/-0.07239 (standard errors across424 paired targets) |
| Existing V3.9 feature replay | First64 scenes agree within4.44e-16 absolute |
| Full scene density | Finite for every continuous and reference scene |

All eight declared pilot gates passed in the first comparison. The reference
list order was then randomized because GalSBI stores galaxy types in blocks;
the scene density uses arbitrary iid order, so an explicit random permutation
matches that representation. This changes neither patch membership nor galaxy
properties/positions. No acceptance thresholds or scientific seeds were tuned.
Final replay job16775946 completed in48s on one inter CPU, passed all gates,
and preserved the count, mark and spatial diagnostics. Final source hashes
were verified and snapshots retained; the comparison plot was inspected.
The full suite gives **331 passed, 3 skipped**; new tests independently check
the normalized Poisson density/count-changing ratio, thinning, empty patches,
support and existing crowding implementation.

This validates an unclustered scene-prior prototype at the tested population
and sample size. It does not test a clustered population, shared overlapping
patches, completeness below the declared population boundary, emulator input
mapping, posterior mixing or shear bias. In particular, the density remains
evaluable without claiming its entire photometric mapping is differentiable.

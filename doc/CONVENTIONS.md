# SBSI conventions

Every reported calibration number must identify the catalogue family, model
set, sample, response estimator, shear component and amplitude, simulation
cases, flow/QMC seeds, and uncertainty estimator.

## 1. Model composition

For every model set,

```text
R_model = R_flow + R_blend.
```

`R_flow` is the response of the joint conditional measurement model.
`R_blend` is the fixed atom-aligned external response contribution and must be
averaged over the same population that passes measured selection.

The active development identity is `V3.5-like`: lambda10 epoch154 joint
four-output flow, trial9 `R_blend`, and the equal-probability three-seed
nine-input coherent-`U` classifier ensemble. It is a model-path identity, not
an inference release or calibration claim.

## 2. Catalogue families

### ConstGold

ConstGold is the antithetic constant-shear evaluation catalogue. Each galaxy is
rendered at `+g` and `-g` with matched noise. Catalogue products can already
encode both-leg detection or usability, so each analysis must state the exact
support it reads. ConstGold outcomes may validate a frozen candidate but must
not select model hyperparameters.

### Half-shear catalogues

Half-shear catalogues pair a zero-shear leg with a forward-sheared leg, usually
at `|g|=0.05` with randomized direction. They are not called constant-shear or
antithetic catalogues.

### Inference prior

The inference prior is a finite scene catalogue with one primary atom per row
and zero-prior-mass neighbours retained for scene context. Its registered
identity lives in `configs/default_catalogue_prior.json`.

Catalogue families are not interchangeable. Results never borrow a response,
selection mask, uncertainty, or case count from another family.

## 3. Population and selection order

For the configured likelihood, apply operations in this order:

1. validate and align explicit catalogue keys;
2. apply the declared parent support and any explicitly declared analysis
   truth cut;
3. evaluate the per-leg usable probability or actual usable flag;
4. draw the joint measured outputs;
5. add the deterministic `R_blend` ellipticity shift;
6. apply measured-output cuts;
7. normalize by the corresponding selected population.

The configured V3.5-like development sample is:

```text
MAG_AUTO < 25.8
FLUX_RADIUS >= 0.75 arcsec
no measured-|e| cut
```

`FLUX_RADIUS` is the convolved SExtractor circular half-light radius. It is not
`0.5*T_PSF` and is not an intrinsic or deconvolved size. Because magnitude and
radius define selection, they remain endogenous joint flow outputs. At
0.2 arcsec/pixel the radius bound is 3.75 pixels; at zero point 30 the magnitude
bound is equivalently `measured_flux_from_mag_auto > 47.8630092322638`.

Unmatched or invalid rows are rejected or dropped and counted. Missing
`R_blend` values are never filled with zero.

### Translating a size-ratio cut into half-light radii

Assume the intended resolution cut is on the PSF-corrected galaxy model:
`T_gal/T_PSF > 0.5`, where `T = Qxx + Qyy` has units of angle squared.
For centered, normalized profiles with finite unweighted second moments,
convolution adds the covariance matrices, so `T_conv = T_gal + T_PSF`.
The cut is therefore equivalent to `T_conv > 1.5*T_PSF` in that convention.
Weighted/adaptive estimates do not generally obey this identity without a
model-specific correction.

For circular Gaussian galaxy and PSF profiles, `Re^2 = ln(2)*T`, giving:

```text
Re_intrinsic > sqrt(0.5)*Re_PSF = 0.707107*Re_PSF
Re_convolved > sqrt(1.5)*Re_PSF = 1.224745*Re_PSF
```

These are radii, while the Gaussian PSF FWHM is `2*Re_PSF`. A Gaussian PSF
with FWHM 0.73 arcsec would thus give thresholds 0.258094 arcsec intrinsic
and 0.447032 arcsec convolved. This is an illustrative Gaussian conversion,
not a measurement of the ConstGold PSF. See the
[ngmix Gaussian conversions](https://github.com/esheldon/ngmix/blob/master/ngmix/moments.py)
and [GalSim moment definition](https://galsim-developers.github.io/GalSim/_build/html/gsobject.html#galsim.GSObject.calculateMomentRadius).

The actual ConstGold `noise.csv` specifies a circular Moffat PSF with FWHM
0.73 arcsec and beta 2.22406805360007. Its analytic untruncated half-light
radius is 0.526773 arcsec; the current renderer's truncation at 4.5 FWHM
gives 0.519272 arcsec before pixel integration. Multiplying either by
`sqrt(0.5)` gives approximately 0.37 arcsec, but assuming this is an intrinsic
Sersic half-light-radius threshold silently assumes equal `T/Re^2` for
galaxy and PSF. That is not generally valid. Half-light radii of arbitrary
convolved profiles also do not add in quadrature.

For an untruncated Sersic profile with true unweighted moments, direct radial
integration gives `C(n)=Gamma(4n)/(Gamma(2n)*b_n^(2n))`, where `b_n` fixes the
half-light radius. Then `T=C(n)*Re^2` for a circular profile, or
`T=C(n)*(1+q^2)*Re_major^2/2` for axis ratio q. At the cut:
`Re_major > sqrt(T_PSF/(C(n)*(1+q^2)))`. The circularized radius is
`Re_circ=sqrt(q)*Re_major`; C(n) is 1.44270, 2.13004, 4.61998, 21.67935 for
n=0.5,1,2,4. These true-moment identities do not convert a Gaussian-fit T
directly into the underlying Sersic truth radius. See the
[Sersic profile definition](https://galsim-developers.github.io/GalSim/_build/html/gal.html#sersic-profile).

BlendEMU's current shape routine fits a PSF-convolved Gaussian galaxy model
and a one-Gaussian PSF model. Its catalogue writer retains only the fitted
shape components, not galaxy/PSF T. An actual T-ratio selection would require
retaining those fitted sizes and modeling that selection. The present
0.5-arcsec truth floor and 0.75-arcsec SExtractor `FLUX_RADIUS` floor are
separate cuts; neither is an established translation of `T/T_PSF > 0.5`.

### Fixed-g0 retraining domain (2026-09-14)

The authorized replacement model domain is anchored once on the zero-shear
measured leg:

```text
S0 = U0
     and MAG_AUTO(g=0) < 25.8
     and FLUX_RADIUS(g=0) > 0.60 arcsec
```

At 0.2 arcsec/pixel the strict radius inequality is
`FLUX_RADIUS(g=0) > 3.0 pixels`. Exact `(case,input_index)` keys carry `S0`
to the sheared training leg; magnitude and radius are not re-evaluated there.
The flow and response emulator use separate target-role key anchors but the
same measured predicate. The flow anchor follows the self-response/secondary
targets. The blending-response anchor follows the simulator's structural
primary role, defined per case by stable
`input_index < floor(N_generated / 2)`; it is not inferred from whether a
shape measurement looks like a failure placeholder. No post-production
truth-property analysis cut is applied; the simulator's finite generator and
target-role boundaries remain the inherited parent support. A response label
is retained only when both response legs have finite ngmix components with
strictly interior shape norm (`g1^2 + g2^2 < 1`). This is label validity, not
an additional population cut.

The usable-event classifier has a different population contract. It is
trained on every object in the half-shear target-role parent, with per-leg `U`
as its label, and receives neither the fixed-g0 measured cut nor an analysis
truth cut. Measurement validity needed to define `U` is not a population
selection. Its inputs exclude `R_blend`, because the replacement response
emulator is trained only on `S0` and would otherwise be extrapolated across
the classifier's full parent.

This retraining domain is not yet the configured likelihood. A flow trained
conditional on `S0` represents `p(y_g | U_g,x_g,S0)`; applying the existing
per-leg measured-cut normalization to it would condition twice. Promotion
therefore requires a distinct fixed-cohort likelihood/configuration and new
caches after model validation.

## 4. Shapes and shear

- `e_truth` is the intrinsic/reduced-shear ellipticity transformed by
  `sbsi.shear_map`.
- `e_flow` is a joint draw from the measurement flow.
- `e_model = e_flow + R_blend * (e_truth(g)-e_truth(0))`.
- measured catalogue ellipticities must be named with their measurement method
  and leg.

Shape response, selected-population response, and total measured response are
different estimands. A result must not call one by another name.

## 5. Usability, detection, and selection

- `D`: raw unique crossmatched detection.
- `U`: an actual usable per-leg shape measurement.
- `S`: the final measured selection event after usability and output cuts.

The V3.5-like classifier estimates `p(U|x)`. It does not estimate `p(S|x)`.
In the configured likelihood, final selection is derived from joint flow
draws. In the replacement fixed-g0 training contract, `S0` is external cohort
membership and is not folded into the classifier target.

## 6. Response estimators

For antithetic legs,

```text
R = [mean(e_plus) - mean(e_minus)] / (2h).
```

For a forward zero/sheared pair,

```text
R = [mean(e_g) - mean(e_0)] / h.
```

Use the same extraction convention on simulation and model sides. Do not mix a
forward numerator with an antithetic denominator or reuse a response measured
on a differently selected population.

Common random numbers are mandatory when differencing stochastic flow
evaluations. The object set, proposal draw, latent prefix, `R_blend` lookup and
selection convention must be paired wherever the estimator assumes pairing.

## 7. Multiplicative bias

SBSI defines

```text
m = R_sim / R_model - 1.
```

Positive `m` means the simulation responds more strongly than the model;
negative `m` means the model response is too large. Report `m` in percent and
state whether uncertainty is a case standard error, bootstrap interval, QMC
error, or a combination.

The 0.3% target requires more than `|m|<0.3%`: truth-population and
measurement-residual components must not hide an opposing cancellation, and a
fresh multi-axis acceptance set remains required.

## 8. Seeds and model selection

Record separately:

- catalogue/simulation cases;
- model-training seeds;
- flow latent or QMC seeds;
- proposal and bootstrap seeds.

Model checkpoints are selected only from their declared response-blind
development score. Response results must not choose a seed, ensemble weight,
temperature, probability offset, cut, or checkpoint.

## 9. Reporting checklist

A complete result states:

- numerical pipeline and likelihood/model identity;
- exact flow, classifier, and `R_blend` hashes;
- catalogue family and cases;
- truth-domain and measured cuts;
- shape kind and shear stencil;
- matched, unmatched, invalid, usable, and selected counts;
- random streams and common-random-number convention;
- `R_sim`, `R_model`, `m`, component decomposition, and uncertainties;
- whether the endpoint was development, reused mechanism screening, or fresh
  acceptance.

Detailed historical conventions and experiment-specific exceptions are
preserved in `archive/research-2026-09-14/doc/CONVENTIONS_FULL.md`.

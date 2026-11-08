# Conditional photometry: completed localization

## Main result

The original NLL photometry flow already selects the wrong mix of neighbour
configurations. The latest continuation reduces that error but leaves a strong
conditional flux-cut deficit. Almost correct global means and total counts
hide opposing errors across contexts. The problem is present in training as
well as validation; a validation-only distribution shift is not required to
explain it. This does not yet distinguish optimization, density-family limits,
or insufficient scene descriptors.

## Population and conventions

Fixed four checkpoints; 500,000 original usable training rows from160 scenes
and500,000 validation rows from40 scenes, original g0/g05 legs combined.
These are ordinary flow-training-domain simulations, not constant-shear
training targets. All quoted uncertainties are whole-scene delete-one
jackknife standard errors with the trained models fixed. Context boundaries
come only from training rows. These are diagnostic populations, not independent
acceptance data. Earlier g0-only numbers differ from the combined-leg values
here. See [contract](rule.md#data-and-exact-replay) and
[project conventions](../../CONVENTIONS.md#9-reporting-checklist).

The central cut is measured/generated radius>3 pixels and magnitude<25.
Passing errors are model minus measured, in **percentage points (pp)**.
Magnitude error is predicted minus measured: positive means too faint.
B denotes the frozen reference-photometry neighbour coefficient used as a
scene descriptor; its selected-mean discrepancy is **not R or shear bias m**.

## Error depends on neighbour configuration

Validation joint passing-rate errors:

| Population | Original NLL (pp) | Latest (pp) |
|---|---:|---:|
| All rows | -0.1337 ±0.0437 | -0.2260 ±0.0434 |
| Lowest neighbour-score fifth | +0.3583 ±0.0581 | +0.1928 ±0.0585 |
| Highest neighbour-score fifth | -1.0205 ±0.1315 | -0.8664 ±0.1319 |
| Close-crowding fourth fifth | -1.3277 ±0.0844 | -1.2931 ±0.0835 |

The highest neighbour-score group has measured passing fraction50.3408%;
original NLL predicts49.3203%, latest49.4744%. The deficit persists at all
nine cuts: NLL -1.103 to-0.712pp; latest -0.903 to-0.655pp.
Training highest-score errors are -1.0332 ±0.1026pp and -0.8818 ±0.1019pp,
respectively. Thus the validation pattern closely repeats the training one.
Neighbour score ranks sensitivity, not simply neighbour density.

## Which photometry component is wrong

An exact decomposition separates radius-marginal, flux-marginal and
radius/flux association contributions to the joint passing error. It uses
symmetric averages of the other marginal probability, so its terms add
exactly; they are not independent intervention effects.

| Highest-score fifth | Radius contribution (pp) | Flux contribution (pp) | Association contribution (pp) | Sum (pp) |
|---|---:|---:|---:|---:|
| Original NLL | -0.2023 ±0.0718 | -0.7418 ±0.0759 | -0.0765 ±0.0591 | -1.0205 ±0.1315 |
| Latest | -0.0591 ±0.0724 | -0.5660 ±0.0765 | -0.2414 ±0.0579 | -0.8664 ±0.1319 |

Flux is the largest term in this important group. Radius passing is much
closer in the latest model, while association still contributes. **Association
here includes remaining context heterogeneity; it is not an identified
per-object conditional copula error.**

The all-row decomposition differs: latest +0.2295pp radius, -0.1440pp flux,
-0.3114pp association sum to-0.2260pp. Reading only the pooled association
term would therefore obscure the localized flux deficit.

## A reproducible close-crowding failure

The fourth training-defined quintile of the existing crowding050 feature,
0.00110424 to0.01880196, shows both too-small and too-faint predictions.
These feature values are not physical neighbour distances. The failure is
not confined to the most extreme crowding bin. This group overlaps the
neighbour-score groups; their effects must not be added as disjoint causes.

| Model, validation rows | Mean radius error (pixels) | Mean magnitude error | Joint passing error (pp) |
|---|---:|---:|---:|
| Original NLL | -0.03158 ±0.00270 | +0.00950 ±0.00088 | -1.3277 ±0.0844 |
| Latest | -0.03587 ±0.00271 | +0.00877 ±0.00088 | -1.2931 ±0.0835 |

Latest training means are -0.03845 ±0.00275 pixels and +0.00903 ±0.00089mag.
Its validation passing deficit decomposes into -0.4656pp radius,
-0.6951pp flux and -0.1324pp association. This reinforces the earlier
[local photometry diagnostic](../flow_photometry_information_20261107/findings.md):
existing inputs contain information about the residual, but adding a scalar
neighbour input alone has not eliminated it.

## Where the selected-neighbour discrepancy originates

Measured minus model mean B among selected rows:

| Model | Training | Validation |
|---|---:|---:|
| Original NLL |  +0.003073 ±0.000303 | +0.002793 ±0.000405 |
| Latest | +0.001838 ±0.000299 | +0.001570 ±0.000406 |

The latest validation point discrepancy is44% smaller. This is compatible
with worse total-count accuracy: correct counts and correct scene composition
are different requirements. It is not a claim that shear bias improved44%.

Using the exact contribution formula in the [contract](rule.md#connecting-photometry-to-response-weighting),
25 intrinsic-radius by neighbour-score cells sum to each global discrepancy,
including every jackknife replicate. Highest-score cells contribute72% of
original NLL and77% of latest net validation discrepancy. Middle three
intrinsic-radius quintiles within that highest-score fifth contribute66%
and79%, respectively. These are descriptive fractions of a signed sum;
cancellations explain why a subset can contribute more than its larger group.
They are not independent causal fractions or percentages of shear bias.
The middle radius range is approximately0.1559–0.3099 arcsec, with B>0.23125.

The largest cell is radius0.1960–0.2388 arcsec and highest B. Its contribution
is +0.000787 ±0.000148 for NLL and +0.000555 ±0.000147 for latest.
Its joint passing deficit is -1.685 ±0.234pp and -1.460 ±0.234pp.
All40 validation scenes contribute rows to these cells.

## Distribution shape also matters

The latest model has almost zero pooled mean magnitude error
(-0.00027 ±0.00049mag), yet a clear conditional passing deficit. A single
photometric zero-point adjustment therefore does not address the pattern.
Independent64-draw observation ranks also show distribution distortions:

- For latest validation, the excess probability that an observation lies
  below all64 draws is +0.071 ±0.016pp for radius and +0.099 ±0.018pp for
  flux. Excess above all64 is +0.090 ±0.024pp and +0.134 ±0.026pp.
  The calibrated null for each end is1/65=1.53846%. Excess on both ends
  is consistent with insufficient tail coverage/underdispersion, but context
  mixtures and asymmetric shape errors prevent reducing it to one width.
- In neighbour-score quintile4, latest high-end excess is +0.435 ±0.049pp
  for radius and +0.524 ±0.046pp for flux. Low-end excesses there are
  only +0.003 ±0.036pp and +0.039 ±0.038pp: the defect is asymmetric.
- Latest highest-score mean flux rank shift is -0.0046 ±0.0012, even though
  flux passing at magnitude25 is underpredicted. Thus a typical-rank shift
  does not determine the CDF error at the target threshold.
- Measured values transformed through the latest inverse photometry flow
  have latent cross moment -0.03619 ±0.00185, compared with the standard
  normal target0. The inverse coordinates mix radius/flux; this is a joint
  density calibration failure, not directly a physical radius/flux correlation.

These checks explain why matching a mean or a handful of selection totals
can leave the conditional distribution wrong. The latest photo NLL is better
than original NLL on these rows (-5.59119 versus-5.58763nat/row), but average
NLL alone does not summarize these localized errors. Endpoints have different
training histories, so this comparison does not isolate a single loss effect.

## What this changes and what remains unresolved

The evidence now identifies a concrete priority: conditional flux passing
in high-neighbour-response configurations, especially intermediate intrinsic
sizes, together with the coupled radius/flux error in a reproducible
close-crowding range. Preserve primary-response gains while addressing this
conditional distribution; removing response regularization cannot remove a
failure already in the original NLL model.

Do not repeat an aggregate selected-neighbour-moment penalty or interpret
these findings as approval for a global flux/radius offset. Those approaches
can correct a summary by moving the wrong population. The earlier failed
[attempts](../training_stage_audit_20261108/prior_attempts.md#measured-input-localization-must-qualify-the-loss-summaries)
remain relevant. The next discriminating diagnostic is conditional CDF/quantile
curves within these identified cells, across nearby magnitude and radius
thresholds, retaining both lower and upper tails. That would separate a local
shift, tail mismatch, and radius-conditioned flux error before choosing a
single-run photometry fit. No new training or loss was launched here.

This audit does not establish that photometry explains the entire physical
neighbour-response or full calibration error. It has localized an existing
photometry/selection discrepancy. Feature sufficiency, optimization and
conditional-density parameterization remain competing explanations.

## Execution and verification

GPU17833751 completed0:0 in3m27s; CPU report17833754 completed0:0 in4m36s;
CPU plot17833774 completed0:0 in6s. Both500k-row splits completed. Exact
photo-only/joint sampling equality; canonical masses and B numerators replay
for all four models and both splits; direct-row/case aggregate agreement;
all marginal decomposition and cell normalization identities passed.
No missing-context rows, exact rank ties or physical flux underflows occurred.
Independent half-bank central joint-rate differences are0.00334pp for NLL
and-0.00131pp for latest globally, and-0.00213/-0.00727pp in highest B:
small relative to the observed deficits. Antithetic draws are not used for
the uniform-rank test. Both rendered figures were visually inspected.

Full machine-readable [report](report.json), generated [tables](results.md#validation-central-cut),
[localization figure](localization.png), and [distribution checks](distribution_checks.png).
Source/model/sample hashes and cache completion receipts are retained.
Broad17788286 remains held; automated follow-ups remain disabled. No models,
production configuration or measurement populations changed.

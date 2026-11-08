# Bright catalogue prior extension — September 25, 2026

## Scope and frozen comparison

The owner supplied the FS2_bright Drive folder and requested every bright row
be added alongside the existing prior. The original prior and all its caches
remain intact. This experiment uses the same V3.6 models and hash-matched
commit773295d runtime as the [current atom maps](figures/inference_atoms_20260925/README.md#results).
It does not integrate V3.9, retrain models, or change defaults.

The first diagnostic is fixed before evaluation: observation rows142230,
3563 and409188; original shear centre; K1024, prefilter131072, M16384,
defensive mixture0.1, fractional flux metric. Compare sampled numerators with
exact sums under **each respective prior**, using every seed8701–8716 for
both priors. The observation sample retains measured radius>0.6arcsec and
magnitude<25.8 with no truth cuts. These are previously inspected cases,
not independent or representative validation. No shear-bias claim follows
from this centre-only diagnostic; selected normalization and shear derivatives
would be required for a full inference comparison.

## Download and catalogue audit

Files are beside the actual existing source in
`/project/ls-gruen/users/zekang.zhang/lsst_sims/FS2/FS2_25876/`:

- `FS2_r_20_IA_Griffith_UDF_i.feather`:389,816,762bytes,
  SHA256`cd7e964ce3fed6d8d8896676e15e97503d31b867f32c7e1f97498f4836bc7f23`.
- `FS2_r_20_README.md`:README renamed to avoid overwriting another source's documentation.

The catalogue has7,780,811 rows, all finite and r<20. Joint
`(halo_id,galaxy_id)` is unique; the `index` column is **not** unique.
All151,828 bright source galaxies in the original catalogue also occur here.
Their sky position, redshift, magnitude, axis ratio and angle agree, but
assigned radius and Sérsic index differ. These are valid additional morphology
realizations: mismatching individual values does **not** establish a different
underlying distribution. The owner expects the same parent distribution;
the intended experiment improves its finite empirical approximation.
`scripts/audit_bright_distribution.py` compares distribution-level effect
sizes, including brightness-conditioned summaries, rather than requiring
individual assignments to agree. No naive iid p-value is interpreted as
evidence for or against a common parent.
Separately,848,650 rows lie outside the old generator's strict radius/Sérsic/domain
bounds; all are retained here. None are silently filtered or clipped.
The catalogue's high-declination footprint must not be treated as a flat
RA×declination rectangle to estimate its density.

## Weights, neighbours and identities

The baseline is the20,231,221-atom uniform-plus-certain-bright prior from
`inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1`.
Let B be the baseline sum of r<20 prior weights, and N_b its269,501 bright
scene atoms. Preserve nonbright weights and give each retained bright atom
and each new bright atom weight B/(N_b+7,780,811). Thus bright mass remains
B, total mass remains one, and the experimental support has28,012,032 atoms.
No empirical offsets, likelihood corrections, or density inferred from the
new sky footprint are used.

The original BlendEMU generator samples source properties with replacement
and draws positions independently. Use that independence to pair each new
intrinsic target with an independently sampled complete neighbour environment
from the original ten uncut scenes20000–20009 (context seed20260925).
This is an empirical product prior, **not** reconstruction of the new galaxies'
actual sky environments. The neighbour population remains the original one.
Each new row gets one environment; uncertainty from environment assignment is
not assessed by repeating inference sampling seeds.

Retain target-independent legacy neighbour flux summaries. Replace target
magnitude, Sérsic index and radius in every emulator pair; recover physical
distance from the donor and rescale by the new target's convolved major
radius. Regenerate all17 classifier inputs at each evaluated shear.
Preserve the original nearest20/10arcsec pair geometry and all-neighbour
flow summaries within7arcsec. Check rescaled features against the canonical
physical rescaling function. Apply the original PA+90deg convention and
circularized-to-major radius conversion. Keep raw catalogue IDs and donor
indices in each shard, with separate globally unique scene-atom IDs.
An overlapping physical galaxy with another morphology/environment remains
a distinct atom; it is not counted as an independent physical galaxy.

For the primary comparison freeze the original proposal coordinate centre,
scale and numerical median scatter floors. Enlarging an oversampled bright
stratum must not silently retune these unweighted statistics. New coordinates
use128 QMC flow draws per atom, matching the existing procedure.

## Distribution-level audit

Completed CPU16696444 preserves source dtypes when reproducing strict
generator bounds and exactly reproduces the previously audited new-domain
mask. Widening float32 q before comparing with0.05 changes boundary handling;
that preliminary audit was corrected and is not the final result.

| Raw bright statistic | Original source | New source |
| --- | ---: | ---: |
| Median r magnitude | 19.349438 | 19.349739 |
| Median Sérsic n | 2.198459 | 2.198459 |
| Median log10 circularized radius | 0.195513 | 0.192671 |
| Mean axis ratio | 0.580398 | 0.581209 |
| Excluded by original generator bounds | 11.2206% | 10.9070% |

Maximum raw marginal CDF differences are0.197percentage points for magnitude,
0.685pp for log radius,0.294pp for Sérsic n and0.300pp for axis ratio.
Redshift differs more,2.153pp, across the different sky footprints. Raw and
matched-domain summaries and five magnitude bins are saved in
`distribution_audit.json`. These are descriptive agreement checks, not proof
of equal joint distributions. They support the intended interpretation of
additional realizations of the same parent population; individual morphology
differences did not warrant the earlier implication of a changed parent.

Of the new catalogue,794,797 rows have n>=6,14,643 have n<=0.5,49,334 have
major radius>=10arcsec and3,023 have q<=0.05 under the original float32
comparison. Categories overlap. All remain in the primary expansion. The
final report measures posterior mass contributed by rows outside the old
generator domain separately from the numerical sampling improvement.

## Reproduction and status

Active implementation: `scripts/bright_prior_extension.py`; scheduler wrapper:
`jobs/job_bright_prior_extension.sh`. Stages: `build`, `evaluate --shard 0..7`,
then `compare`. GPU evaluations use at most two concurrent workers.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_bright_extension_20260925_v1`.
Contains immutable download hashes, catalogue audit, source snapshot, per-shard
prior and evaluation receipts, and full exact density terms for reproduction.
Each evaluation verifies source/model/runtime hashes. The baseline seed8701
must reproduce the previously saved exact diagnostic.

Download/audit job16696367 completed. Initial construction16696388 passed
the physical rescaling checks but GPU startup16696392 caught malformed NPY
headers from NumPy-scalar shape dimensions. Cancelled that array and its
dependent comparison16696395. Corrected dimensions to Python integers and
added an explicit file-reopen check. Rebuild16696407 completed in2m19s;
all eight shards and canonical pair-rescaling/file-reopen checks passed.
Total bright probability is0.00192586465478398; each new/retained bright atom
has weight2.392285733501981e-10. GPU array16696409 uses at most two V100s;
comparison16696411 depends on successful completion of all eight shards.
All eight GPU shards completed. First comparison16696411 and report16696424
passed numerical replay but inadvertently allowed the coordinate constructor
to refit the derived fractional metric. Those results are preserved externally
with the `implicit_metric_refit` suffix and are not the planned controlled
comparison. Corrected comparison16696478 freezes the derived metric and checks
exact coordinate-standardization identity; report16696481 follows it. All
valid exact likelihood terms are reused. Corrected comparison16696478 finished
in1m30s and report16696481 in9s. All three independent log-sum/draw-replay checks
passed; baseline seed8701 reproduces the original diagnostic. No jobs remain.

## Results and decision

All7,780,811 supplied rows are retained in the28,012,032-atom experimental
weighted union. `combined_prior.json` at the external root explicitly binds
both manifests and the baseline bright-weight adjustment. The original prior
remains the default. [Reports and figure](figures/bright_prior_20260925/README.md#fixed-observations-and-16-sampling-seeds)
include16 seeds per observation, exact terms, source snapshots, validation and
the distribution audit.

| Observation | Existing relative numerator RMSE | Expanded RMSE | Existing posterior atom ESS | Expanded ESS |
| --- | ---: | ---: | ---: | ---: |
| 142230 | 0.000311 | 0.848967 | 1.151 | 1.641 |
| 3563 | 0.927664 | 1.491547 | 1.220 | 4.727 |
| 409188 | 1.013688 | 0.879673 | 1.742 | 2.657 |

RMSE is measured against each prior's own exact centre numerator across the
same16 seeds. These are **not shear-bias values**. Effective posterior support
increases modestly, but this does not yield reliable integration: two cases
have worse RMSE, and all three remain unstable. The smaller third-case RMSE
does not establish convergence; its mean estimated/exact ratio is0.1305.

For row142230, original atom14591211 still carries77.62% of the expanded
posterior. It was in the old exact list, but is absent from the new exact list
and the seed8701 sample. The new list contains994 new atoms out of1024 yet
only0.0396% of exact posterior mass. The former exact list would contain83.34%
under the expanded prior. This demonstrates a candidate-ranking failure as
support grows, beyond the implicit metric-refit confound corrected above.

New rows outside the original generator domain contribute0.0381%,2.5173%
and1.3497% of posterior mass respectively. They do not account for the large
integration failure. Do not infer a changed parent distribution from the
different individual morphology realizations.

Decision: retain the new catalogue and experimental prior, but do not promote
this combination as an inference improvement. Improve candidate ranking and
verify integration across a representative set before a full shear-calibration
claim. No selected-normalizer/stencil fit or0.3% result was produced here.
The malformed temporary array copies were removed after retaining their
failure receipts; all valid data and both comparison variants are preserved.

## Owner-requested 8k exact plus 8k sampled test

The owner then requested8192 exact+8192 sampled atoms. GPU16696513 and report
16696518 completed using the same three observations,16 seeds, fixed metric,
prefilter131072, floors and model artifacts. This reduces total budget from
17408 to16384 while increasing the deterministic stratum. All six exact-sum
and saved-draw replay checks pass; original exact sets are subsets of the new
ones. [Figure, metrics and validation](figures/inference_8k8k_20260925/README.md#results).

| Observation | Original-prior RMSE at8k+8k | Expanded-prior RMSE at8k+8k | Expanded exact posterior mass |
| --- | ---: | ---: | ---: |
| 142230 | 0.00930% | 1.861% | 97.319% |
| 3563 | 0.04321% | 267.712% | 0.656% |
| 409188 | 0.00491% | 87.365% | 1.634% |

Percentages are relative likelihood-numerator RMSE across seeds, not shear
bias or worst-case error. All three original-prior cases sum over99.8% of
posterior mass exactly. Expanded row142230 now recovers its dominant atom;
the other two still miss theirs.8k+8k is promising for original-prior
integration, but insufficient for the expanded prior under current ranking.
Three inspected cases do not establish population-wide convergence. Defaults
remain unchanged and there is no new shear-calibration measurement.

## Simpler atom-selection scores

GPU16696664 compared current Gaussian re-ranking with nearest-location,
prior/usability-weighted common-kernel distance, and unfloored Gaussian scores.
The existing sampling distribution and draws stay fixed to isolate exact-set
selection. [Full comparison and variance derivation](figures/inference_simple_scores_20260925/README.md#results)
include both budgets, both priors, all sixteen seeds, and exact repeated-draw
variance calculated over the whole finite prior.

At K1024/M16384, nearest-location reduces expanded-prior numerator RMSE from
84.897%/149.155%/87.967% to2.163%/7.006%/3.421% for rows142230/3563/409188.
Their theoretical relative sampling SDs are still3.717%/20.124%/8.084%; rare
importance weights make sixteen seeds insufficient to characterize the tail.
The raw-scatter Gaussian is best on the original-prior diagnostic examples
but less reliable than nearest-location on the expanded prior. The weighted
common kernel behaves similarly to nearest-location here.

Nearest-location is the simplest candidate for subsequent representative
validation: remove re-ranking and retain the existing frozen measurement
metric. It is a retrieval rule, not a calibrated likelihood; actual prior
weights remain in the estimator. No production change. The prefilter itself
excludes up to4.146% posterior mass on these expanded-prior cases, so ranking
alone cannot settle convergence. No new random proposal or shear fit was
tested; validation must include representative rows and the shear stencil.

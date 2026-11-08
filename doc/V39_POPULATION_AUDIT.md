# V3.9 prior population and integration audit

## Question and scope

On 2026-09-27 the owner asked whether the expanded bright prior represents
the wrong population and requires reweighting. This audit separates three
questions: whether the prior matches the unselected simulation population,
whether the matched-input mock has that population, and whether likelihood
integration is numerically reliable. No production weights, models, samples,
selection cuts or inference defaults are changed.

The scientific target remains constant-shear image measurements with measured
radius >0.6 arcsec and magnitude <25.8, without truth-property cuts on the
measurement sample. Histogram discrepancies below are probability differences,
not multiplicative shear bias; see [conventions](CONVENTIONS.md).

## Sources and reproducibility

The external evidence root, relative to
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior`, is
`inference_v39_population_audit_20260927_v1`.

- `audit/population.json` and `population_histograms.npz`: all 100 unselected
  ConstGold input cases 40–139, the original weighted prior, the expanded
  weighted prior, and a domain-matched counterfactual.
- `audit/influence.json`: saved 8,192/16,384-draw per-observation moments from
  the [expanded comparison](V39_MATCHED_INPUT_MOCK.md#completed-expanded-prior-result-on-september27).
- `profiles/`: exhaustive atom sums for two influential observations at the
  existing expansion point and its two g1 neighbours.
- `crowding/`: separate comparison of existing neighbour-feature caches.

Active diagnostic sources are `scripts/audit_v39_population.py` and
`scripts/profile_v39_population.py`; executed snapshots are preserved under
the evidence root with hashes and execution receipts. Input catalogues and
prior arrays are checked against their saved SHA256 identities. CPU audit
16730323 completed successfully in 1m35s; its Python work took 84.1s. Full
tests 16730371 passed: 319 passed, 1 skipped, no regression/snapshot failures.

## Original prior versus the actual unselected parent

The audit scans **69,956,785 actual input rows** before measured selection.
Both configurations use the same FS2 catalogue and generator loader. The
loader SHA256 matches the prior generation manifest:
`80dc9d028d6bf347e4397d4f9b58844cbd5bc944709b7fc5bdf6e268bc947149`.
The loader enforces strict 0.05 < q < 1, 0 < r < 29,
0.01 < semi-major Re < 10 arcsec, and 0.5 < n < 6. No audited actual input
lies outside the morphology domain.

| Weighted prior versus unselected inputs | Original | Expanded | Domain-matched counterfactual |
| --- | ---: | ---: | ---: |
| Bright probability outside generator domain | 0% | 10.5418% | 0% |
| Mean n among true r <20 | 2.49576 | 2.83825 | 2.49849 |
| Bright n binned CDF discrepancy | 0.28855 pp | 9.87287 pp | 0.61576 pp |
| Bright four-property joint histogram TV | 1.6168% | 10.4729% | 1.9900% |

The actual unselected bright mean n is 2.49207. For the original prior,
the largest full-population binned marginal CDF discrepancy over magnitude,
circularized radius, n and q is **0.02382 percentage points**. Its bright
probability is 0.192586%; the actual parent contains 134,530 bright rows.
The full-population four-property joint histogram TV is 0.19348%.

These are descriptive comparisons of finite, related samples, not independent
significance tests or proof of equality of the full latent distribution.
The documented historical NumPy/pandas runtime difference also means the
generator pools are not bitwise identical; see
[runtime compatibility](ENVIRONMENT.md#historical-scene-prior-runtime-compatibility).
Nevertheless, these checks give no evidence for a large intrinsic-population
mismatch in the original prior.

Agreement at this level is not sufficient to certify0.3% shear calibration.
For example, mean intrinsic e1 is+5.905e-6 in the original weighted prior
and-6.007e-5 in these actual parent realizations. That6.60e-5 difference is
not itself a shear-bias estimate or proof of systematic mismatch, but it
illustrates why finite-prior uncertainty needs checking at the target
precision. Do not subtract a fitted shape offset; use independent prior
realizations/integration checks to quantify the effect.

Target-role and other-role unselected inputs also agree closely in these
properties: their largest marginal CDF discrepancy from the combined parent
is about 0.01061 percentage points. The generator samples properties and
positions independently; role assignment does not use different intrinsic
population samplers. This does not establish equal model accuracy by role.

## Neighbour-feature population check

CPU audit16730457 completed in42s. It compares all **27,982,713 unselected
target-role inputs** in cases40–119, including53,843 bright inputs, with
the three close-crowding features actually used by F8/C8. Existing feature
cache hashes are verified; scene geometry is not recomputed.

For the original prior the largest full-population neighbour-feature binned
CDF discrepancy is **0.03609 percentage points**. Among bright inputs it is
0.19917 percentage points. The coarse joint histogram of magnitude and all
three crowding features has TV0.19020% for the full population and0.53016%
for the bright population. The domain-matched expanded bright counterfactual
has maximum marginal discrepancy0.14921 percentage points and jointTV0.48591%.

Thus this check does not reveal a large marginal crowding mismatch either.
It covers target-role inputs in80cases and selected coarse joint bins,
not the full eight-dimensional joint density or model accuracy.

## Specific correction justified for the extra bright support

The expansion already preserves the original total bright probability,
**B = 0.00192586465478398**, and all nonbright weights. It does not simply
assign equal weight to every old and new atom. The error is within the bright
stratum: **848,650** of the 7,780,811 added rows are outside the domain used
to generate these image simulations. Their current probability is
0.000203021328774, or 10.5418% of B.

A minimal, population-defined counterfactual is:

1. Give those outside-domain added atoms zero weight for this ConstGold target.
2. Preserve B and redistribute it uniformly across the 269,501 old bright
   atoms plus 6,932,161 valid added atoms: 7,201,662 in total.
3. Keep every nonbright weight and every selected measurement unchanged.

The resulting weight per valid bright atom is 2.67419472725e-10, a factor
1.11784085396 above the current expanded bright weight. This corrects the
generator support; it is **not a truth-property cut on the observed sample**.
Newly generated galaxy properties need not equal those of individual old
galaxies. The relevant test is the population after the same generator
conditioning, not an object-by-object equality test.

Only counterfactual histograms and likelihood numerators are evaluated here.
An inference experiment with these weights must also recompute the selected
normalization and update the proposal and its importance probabilities
consistently. Broad fitted magnitude/size/n/shape weights are not yet justified
by this evidence, and must not be tuned to recover the injected shear.

## Why the matched-input mocks have a different target distribution

Let p(z) denote the unselected parent and a(z; g0) the probability that the
original image measurement passes its selection at the injected shear.
The inputs associated with the selected actual20k cohort follow, in expectation,

`r(z) ∝ p(z) a(z; g0)`.

The matched mock then performs a fresh C8 usability draw, F8 measurement draw
and measured selection. Its appropriate selected density is proportional to

`integral r(z) P_model(U | z, g) f_model(y | z, U, g) dz`,

normalized by the model selection probability integrated against **r**.
The ordinary actual-data likelihood instead integrates against **p**, with
one corresponding selection normalization. The mock distribution therefore
does not become an ordinary population-closure sample merely because the
forward draw is correctly implemented.

The observed shift is substantial: mean input magnitude is 26.40272 in the
unselected parent, 24.73266 in the actual selected20k inputs, and 24.50544
in the 16,314 retained matched mocks. True-bright fractions are approximately
0.1923%, 0.72% and 0.8827%, respectively. These are expected selection shifts,
not evidence that the actual-data prior should be fitted to the selected
input histogram. Doing that and retaining the ordinary selected likelihood
would generally count selection incorrectly.

For a clean population-closure test, draw new mocks from an unselected parent
and apply usability and measured selection once. To study this exact cohort
conditionally, specify its conditional input distribution and normalization
explicitly. Either is distinct from accepting the existing mock Newton step
as a calibration result.

## Draw instability is concentrated in bright observations

At fixed prior, changing the integration draw count does not change the
physical parent population. Therefore a population mismatch alone cannot
explain the severe 8k-to-16k numerical changes.

Projecting each observation's information along the smallest-eigenvalue
direction of the final summed matrix gives:

| Expanded sample | Measured-bright observations | Their projected information | All observations |
| --- | ---: | ---: | ---: |
| Actual20k | 135 | -999,244.60 | -995,018.53 |
| Matched mocks | 134 | -227,704.62 | +112,803.98 |

For actual data, the measured-bright group accounts for 100.42% of the net
negative information; positive contributions elsewhere partly cancel it.
The highest-change actual observation, row1958, contributes -502,037.35
along this direction. Its sampled weight ESS is 1.124 and its largest
sampled draw carries 94.24% of the weight. The highest-change mock row4460
has sampled ESS 2.353 and largest sampled weight 64.47%.

These diagnostics show poor integration support for influential rows, but
sampled-draw ESS is not distinct posterior-atom ESS. Exhaustive sums below
separate missing posterior mass from an intrinsically narrow finite-prior
posterior. No influential observation is removed from inference.

The selected normalization is unchanged across draw counts for a fixed
prior, so it cannot cause that draw-count drift. Its g1 log-mass curvature
is-1.35163 per observation for the expanded prior, versus-1.35227 for the
original. The exhaustive selected profiles still include that existing
normalization, with its finite-QMC limitations.

## Exhaustive likelihood probes

The two probes were chosen after inspecting the largest absolute projected
8k-to-16k information change in each sample: actual row1958 and mock row4460.
They are diagnostic, not a representative acceptance sample.

All 28,012,032 atoms are summed at g1 = 0.009836059884222188 ±0.001 and
the central value, holding g2 = -0.00029219982162802493. Each probe retains
its original four-observation density group; density atom chunks remain4096.
Three GPU workers partition all atoms without gaps. Contributions are split
into old nonbright, old bright, valid new bright and outside-domain new bright.
The original prior and domain-matched numerator are reconstructed from those
same exhaustive contributions, without drawing another proposal.

All three GPU workers16730364[0–2] and CPU reducer16730370 completed
successfully. Each worker's numerical loop took635–641s; total GPU job
times were about14minutes including loading. CPU16730526 also verified the
original-prior reconstruction against all20,231,221 old weights: every old
bright weight is7.14603899349e-9, the exact bright multiplier is29.8711767303,
and nonbright weights are unchanged.

Here information means **minus the g1 second derivative**; positive values
indicate local concavity along g1. These are per-observation quantities,
not full-sample information or a shear estimate.

| Probe | Quantity | Importance8k | Importance16k | Exhaustive expanded |
| --- | --- | ---: | ---: | ---: |
| Actual row1958 | g1 score | +225.65 | -705.55 | +141.56 |
| Actual row1958 | g1 information | -35,373.35 | -476,790.52 | -158,392.95 |
| Mock row4460 | g1 score | -91.00 | +591.53 | +97.91 |
| Mock row4460 | g1 information | -30,543.77 | -233,919.29 | +165,073.89 |

For the mock, even the sign of the sampled curvature is wrong at both draw
counts. For the actual probe, the16k score has the wrong sign, while negative
curvature persists in the exhaustive sum. Population mismatch cannot account
for disagreement between importance and exhaustive integration of the same
fixed likelihood. This establishes a consequential numerical integration
failure for these influential rows; it does not establish its frequency
across all observations or determine the correct full-sample shear.

The exhaustive posterior is also sparse:

| Probe | Distinct-atom ESS | Largest atom fraction | Outside-domain atom mass |
| --- | ---: | ---: | ---: |
| Actual row1958 | 5.608 | 39.57% | 13.284% |
| Mock row4460 | 1.662 | 76.70% | 3.431% |

These are posterior likelihood-mass fractions, distinct from the prior bright
mass fraction. They show that poor sampled ESS is not solely a proposal issue:
even an exhaustive sum is supported by very few atoms. Sparsity alone is not
proof of bias, but off-prior closure and prior-resolution sensitivity must be
tested. Prior-generated mocks are guaranteed to contain their generating atom;
separate catalogue-input mocks are not, so success on the former does not
validate interpolation between prior atoms.

The original-prior exhaustive g1 information is-79,970.57 for the actual probe
and+44,799.32 for the mock. Under the domain-matched counterfactual, numerator
information is-8,834.75 and+170,036.63, respectively. For comparison, the
expanded numerator information before its selected normalization is
-158,391.60 and+165,075.24. The domain correction substantially changes the
actual probe's curvature and nearly leaves the mock's curvature unchanged;
it is not a universal numerical cure.

Original/expanded selected profiles use the existing64-QMC-draw normalization,
not an analytic selection integral. The domain-matched profile is
**numerator-only** until its normalization is recomputed. Fixed h=.001 and
density floating-point limitations remain. These three-point probes neither
locate an optimum nor establish convergence as h decreases.

## Decision boundary

The evidence supports the following sequence:

1. Make a separate domain-matched expanded-prior candidate, preserving bright
   total probability and all measurements. Recompute its selected normalization
   and proposal probabilities; do not alter the retained production result.
2. Validate bright integration against exhaustive references before running a
   larger inference sample. The current8k/16k estimates fail on these probes.
   Test more predeclared bright/near-bright observations, independent proposals
   and both shear directions; keep kernels and common random numbers fixed.
3. Check finite-prior resolution using independently generated denser valid
   support and off-prior mocks. More repeated draws cannot create missing
   latent support. Distinguish proposal improvement from a change to the
   finite population model.
4. Generate a small unselected-parent mock cohort, apply model usability and
   measured selection once, and establish numerical convergence before
   interpreting shear calibration. Keep the existing matched-input cohort as
   a separate conditional diagnostic.

No broad density-ratio fit, empirical offset, favorable-draw-count choice,
bright-observation removal, accepted shear or robust0.3% claim follows.

## Domain-matched pilot authorized on September27

The owner authorized the next steps. The isolated candidate is
`inference_v39_domain_matched_pilot_20260927_v1` under the same external base.
`scripts/run_v39_domain_matched.py` is the active source; a hash-pinned
snapshot, protocol and execution identity are retained there. Production
weights and prior comparison outputs remain unchanged.

Initialization16730721 passed:848,650 added rows receive zero mass,
7,201,662 valid bright atoms share the original bright total, and every
nonbright weight is unchanged. Zero-weight rows stay aligned with response
caches but are excluded from proposal support. Reusing cached C8 probabilities
is valid because they condition on each atom, not the population weights.
The selected normalization is recomputed using the same64QMC draws per atom,
the same shard seeds and all10cached nodes. Each forward draw is reduced with
both the old and new weights; old-weight shard masses must replay within
absolute2e-13 before any sampler evaluation proceeds.

The predeclared focal cohort preserves four original groups,16evaluation rows:

| Sample | Focal row | Measured magnitude | Selection |
| --- | ---: | ---: | --- |
| Actual |1958 |18.2211 |Previously identified failure |
| Actual |5359 |19.8794 |Fixed random seed2026092711 |
| Matched mock |4460 |18.4182 |Previously identified failure |
| Matched mock |3192 |18.0055 |Same fixed random stream |

No truth property or inferred shear is used to select the two random bright
rows. Other group members retain their original numerical role; padding is
not promoted to a scientific observation. This is a bright integration pilot,
not a representative population sample.

Three candidate-selection arms retain K=1024 and the same tilted full-support
complement, delta0.1, original absolute IDs and draws512–16384:

1. Existing Gaussian reranking within131,072 nearest-location atoms.
2. Exact centre likelihood contribution ranking within that location pool.
3. Exact centre contribution ranking within its union with the131,072 best
   whole-catalogue Gaussian-proxy atoms.

The exact ranking score is log[prior weight × C8 probability × conditional
measurement density], with no fitted proxy factor. The same candidate stratum
is retained across the shear stencil. Proposal seeds8701,8702 and8703 are
paired between arms. Exhaustive sums over all28,012,032 aligned rows at all
nine stencil points measure score, full2×2information, posterior atom ESS,
and the likelihood mass captured by each pool and final candidate set.
Original four-observation groups and4096density chunks are preserved.

The numerical pilot gate is predeclared: every focal row, both8k/16krungs and
all three seeds must agree with exhaustive scores within0.1+0.001×|reference|
and each information entry within10+0.001×|reference|. These tolerances are
numerical screening criteria, not0.3% shear-bias acceptance. A failed gate
prevents treating larger-sample inference as a validated continuation; it
does not trigger removal of a difficult observation or selection of a lucky
seed. Off-prior closure and independent support-resolution tests remain next
steps after examining this evidence.

Full tests16730722 passed317, skipped3 without artifact roots configured;
rerun16730766 with those roots passed319, skipped1. Array16730737[0–3]
requests exactly four persistent
one-GPU workers on32GBV100s in inter, followed by CPU reducer16730742.
The first submission16730727 was cancelled during startup because the node
list requested both hosts for each task; no completed scientific stage was
used from it. The corrected launcher explicitly requests one node per worker.
Startup and normalization replay are monitored before yielding to the long
exhaustive stage. Estimated runtime is roughly1.5–2hours after startup,
subject to measured throughput;4hour limits allow for shared-node variance.

At startup all four workers passed their first normalization shard's
old-weight replay **exactly**, maximum difference0.0, and entered the second
shard. All four allocations have one node and one GPU, three on th-cl-nv01
and one on th-cl-nv02. No failure receipt exists. CPU16730778 will check the
two known probes' new exhaustive numerators against the previous independent
component reconstruction after reduction. Status-only audit16730779 is
scheduled about two hours after submission; it writes an external report
and does not send a notification or automatically promote an inference result.

## Fresh unselected-parent mock preparation

The independent next cohort is under
`inference_v39_unselected_parent20k_20260927_v1`; active source is
`scripts/run_v39_parent_mock.py`, also frozen with its own protocol hash.
CPU16730765 completed in1m43s. It uniformly selected20,000 unique input rows
without replacement from all69,956,785 unselected rows in cases40–139,
then reconstructed their F8/C8 contexts and full-scene disk pair features.
No truth-property or measured-selection cut was added. All20,000 identities
matched;9,982 have target role and10,018 have other role. Four identities
overlap the previous selected20k sample by chance; they are retained rather
than conditioning the population by excluding them.

Generation16730768 depends on successful preparation and completion of the
four-worker GPU array, so it cannot exceed the four-GPU ceiling. It performs
one C8Bernoulli draw and one F8/disk draw per usable input at injected(0.02,0),
then applies the measured radius>0.6arcsec and magnitude<25.8 cuts once.
Seeds are inputs2026092713, detection872101 and flow872102. This means
**20,000 input attempts**, with retained count reported and no top-up.
Inputs are never inserted into the inference prior. Generation proceeds as
an independent diagnostic even if the sampler pilot fails; inference on this
cohort remains pending numerical evidence. These are existing finite scenes,
not an independent image-simulation acceptance set.

## Completed domain-matched pilot on September27

All four16730737 workers completed successfully in55m04s–55m07s, reducer
16730742 in26s, and independent replay16730778 in1s. All28normalization
shards reproduce the old-weight masses exactly. At the unchanged expansion
point, selected mass is0.2595175919582845. The two known probes' directly
reweighted exhaustive numerators match the earlier component reconstruction
within2.85e-14in log space. Worker receipts and all saved array hashes were
verified at the owner's status check. No failure receipts or pending pilot
jobs remain.

The proposed exact likelihood ranking materially improves local mass coverage,
but **all three arms fail the predeclared numerical gate**. Every focal
observation fails at least one moment criterion in each of its six seed/rung
checks. Minimum captured likelihood mass over all nine shear points is:

| Focal observation | Gaussian top1024 | Exact-ranked union top1024 | Entire screening union |
| --- | ---: | ---: | ---: |
| Actual1958, known failure |0.000459% |93.4727% |94.4472% |
| Actual5359, fixed random |0.457392% |98.2813% |99.2961% |
| Mock4460, known failure |0.0000239% |98.7186% |98.7884% |
| Mock3192, fixed random |98.7574% |99.9598% |99.9906% |

The first column measures only the exactly summed1024-atom stratum; the
estimator also samples its complement. Tiny local coverage exposes almost
the entire integral to that noisy complement, and does not mean the full
estimator deliberately discards that mass.

For actual1958, the16k g1 score error falls from-245.90,-446.72,-362.40
across the three Gaussian seeds to+5.17,+4.80,-2.56 under exact union ranking.
For mock4460, it falls from-92.49,-111.32,-110.41 to+0.459,+0.409,+0.472.
These are large improvements, but curvature errors remain above the fixed
gate. Neither lower scatter nor a more favorable estimate constitutes
numerical convergence.

The entire screening union still misses5.55% of actual1958 likelihood mass
at its worst node. Summing that entire union without a complement would
leave a g1 score error+10.36 and information11error+2525.70 for that probe.
Therefore increasing K only within this pool cannot eliminate the missing
contribution. `restricted_pool_errors.json` records exact deterministic
pool errors separately from the sampled complement; no observations are
removed and no gate is relaxed.

Corrected-prior exhaustive distinct-atom ESS values are4.402,2.168,1.552,
3.092 in the above row order. Finite-prior resolution remains a separate
issue even after numerical integration is repaired.

Fresh mock generation16730768 completed in15s:20,000unselected inputs,
8,972usable, **5,130retained**, including44with measured magnitude<20;
3,842usable draws failed the measured cuts. There was no top-up, prior-atom
insertion or input measurement selection. No shear inference has yet been
accepted on this new cohort.

## Exact whole-prior ranking follow-up

Under the existing authorization to repair bright integration, the next small
pilot removes the screening-pool restriction. Active source is
`scripts/probe_v39_full_rank.py`, frozen under
`inference_v39_fullrank_pilot_20260927_v1`. It uses the same four groups,
domain-matched prior, normalization, draw ladder, proposal seeds and numerical
gate. No new scientific observation or model is introduced.

Each worker evaluates every aligned atom at the centre once, retaining the
highest exact contributions. Candidate counts1024,4096,16384 are predeclared.
Each candidate set is summed exactly and its complement sampled with the
existing full-support tilted proposal. Its mass and moments across all nine
nodes are compared with the completed exhaustive reference. The centre sum
and selected atom densities must replay within absolute1e-7in log space.
This changes candidate support without fitting another proxy or tuning to
the injected shear.

Array16731606[0–3] uses at most four one-GPU V100 workers; reducer16731607
depends on it. Runtime is expected to be much shorter than the previous
pilot because normalization and nine-node exhaustive references are reused.
This remains an integration diagnostic, not a production change or0.3%
shear acceptance.

Startup succeeded on four32GBV100s, one per worker on th-cl-nv01. At6minutes
elapsed, each worker had entered the exact centre scan, with roughly5.2million
atoms scored and no failure receipts. Current throughput suggests another
15–20minutes including candidate comparisons. Status-only audit16731657 is
scheduled in about two hours; it records results without promoting a sampler
or sending an owner notification.

## Completed whole-prior ranking pilot on September27

All four workers16731606[0–3] completed exit0 in17m24s–17m28s;
reducer16731607 completed in19s. All worker and reference artifact hashes
were verified, and all four exact-centre density/sum replay checks passed.
The refreshed status receipt records completion without promoting a sampler.

With16,384 exact candidates, both mock focal rows pass all three seeds at
both8,192 and16,384 complement draws. Neither actual-data focal row passes
every check. Results below report the minimum exact-candidate mass coverage
over nine shear nodes; the sampled complement is additional to that mass.

| Focal observation | Coverage at K16,384 | Passing checks out of six |
| --- | ---: | ---: |
| Actual1958, known failure |99.629465%|0|
| Actual5359, fixed random bright |99.803972%|3|
| Mock4460, known failure |99.915333%|6|
| Mock3192, fixed random bright |99.996588%|6|

All score components pass at K16,384; the remaining failures concern
information/curvature. For actual1958 at16,384 draws, g1 score errors across
the three seeds are+0.18377,+0.05851,+0.16615 against exact266.92208.
The largest absolute information-entry errors are48.52,34.44,54.34; the gate
checks each entry against its own unchanged10+0.001|exact| tolerance.
Thus this is a substantial integration improvement, but the overall pilot
still fails at each candidate count1024,4096,16384.

A saved-array diagnostic, `tail_diagnostic.json` in the full-rank root,
finite-differences candidate-only minus exhaustive log numerators. The
common selection normalization cancels. Dropping the remaining tail at
K16,384 would change actual1958's g1 score by+0.66258 and its11information
entry by+172.77; for actual5359 the changes are+0.25545 and+48.15.
These deliberately truncated calculations are diagnostics, not a proposed
estimator. Small omitted mass does not ensure accurate curvature.

The next numerical target is the complement contribution for the actual
probes, retaining the same seeds, exhaustive references and acceptance gate.
Possible repairs must be compared on moment errors, not mass coverage alone.
Only after numerical agreement should the fresh unselected-parent mocks be
used for population closure. Both passing mock probes here belong to the
older doubly selected cohort; they are not a new mock shear result. Finite
prior support remains a separate unresolved issue even with exact sums.
No production change, accepted shear result or robust0.3% claim follows.

## Exact-centre conditional-complement pilot

Owner authorized continuation after the whole-prior ranking result. Frozen
root: `inference_v39_exacttail_pilot_20260927_v1`. Sources are
`scripts/probe_v39_exact_tail.py` and `scripts/v39_exact_tail.py`; the latter
has focused algebra/support/random-stream tests in
`tests/test_v39_exact_tail.py`. Production inference remains unchanged.

The preceding pilot fixes the main candidates but retains the Gaussian
whole-prior proposal for complement draws. This follow-up changes both its
score and its conditioning: every draw now comes from outside the exact
candidate set, using the exact likelihood at the fixed centre. Let
`c_j(g) = pi_j P(U|j,g) f(y|j,g)`, and let C be the same frozen top-K set.
For each observation use

```
q(j) = 0                                      for j in C
q(j) = 0.9 c_j(g0)/sum_outside c_j(g0)
       + 0.1 pi_j/sum_outside pi_j             otherwise

Z_hat(g) = sum_C c_j(g) + mean_draws[c_j(g)/q(j)]
```

The defensive component retains every positive-prior atom outside C, even
if its centre likelihood is zero. The exact part and complement are disjoint.
This preserves the expectation of the numerator integral; it does not imply
unbiased finite-draw logs, scores or curvature. Those still must pass the
unchanged exhaustive-reference checks. Actual floating-point CDF bin widths
are used as the reported proposal probabilities, and support loss from
rounding raises an error.

Retain the same four original observation groups, corrected prior,
normalization, centre, step h0.001, candidate sizes1024/4096/16384,
seeds8701–8703 and draw ladder512–16384. Every seed must pass both8k/16k
score/information gates on all four focal rows. Original per-object uniform
column1 is reused, and the same draws are held across all nine shear nodes.
Density batches remain four observations and4096 atoms.

Each worker rescans the full prior once at the centre, validating the total
and all16,384 retained contributions against the preceding run within1e-7
in log space. Unlike that run, all centre contributions are retained on disk
for reuse, about0.90GB per worker. No full reference sum at displaced shear
points enters the proposal. This is a numerical repair experiment requiring
an expensive centre scan, not yet an efficient large-sample pipeline.

Ruff and AST checks passed. CPU test16731872 passed37 tests in22.11s, including
the existing catalogue-null tests. Workers16731880[0–3] depend on successful
tests; reducer16731881 depends on the workers. Four one-GPU V100 allocations
in inter respect the combined four-GPU ceiling and45minute worker limits.
The first queued array16731873 and reducer16731874 were cancelled before
allocation to fix the launcher's ranking-reference path; neither ran science.
Status-only audit16731875 is scheduled for about two hours. Numerical success
would justify the next closure diagnostic, not a0.3% shear claim.

Startup was checked through first meaningful density progress: all four
workers see one32GB V100 and had scanned1,064,960 atoms at4m05s elapsed,
with no failure receipts. The observed scan rate is about30s per1.05million
atoms after the first compilation; estimated another15–20minutes including
the proposal comparisons. These are runtime observations, not numerical
acceptance results.

## Completed exact-tail pilot and fresh-parent first pass

Workers16731880[0–3] completed exit0 in17m35s–17m43s; reducer16731881
completed in20s. At K16,384 **all24 predeclared focal-row/seed/rung checks
pass**, with no threshold or seed change. K4096 passes22/24 and K1024
passes6/24, so neither smaller candidate size passes the combined gate.

| Focal observation | Maximum absolute score-component error | Maximum absolute information-entry error |
| --- | ---: | ---: |
| Actual1958 |0.010354|20.6443|
| Actual5359 |0.002558|1.4313|
| Mock4460 |0.001036|1.7580|
| Mock3192 |0.000115|0.1105|

These maxima cover all three seeds and both8k/16k draw counts. The gate is
entrywise, so the information error20.6443 for actual1958 must be compared
with its corresponding cross-entry tolerance21.3091. This passes, but is
close to that tolerance. The claim is numerical agreement on four development
probes, not general convergence or a recovered shear bias. Maximum floating-
point CDF probability discrepancy is1.67e-16, with all support checks passing.
Worker and exhaustive-reference hashes were verified; the next scheduled
initializer additionally verified all four retained0.90GB centre-array hashes.

Continue under the owner's authorization with
`scripts/run_v39_parent_inference.py`, frozen in
`inference_v39_parent5130_exacttail_20260927_v1`. This uses the fresh normal-
draw unselected-parent sample:5,130 scientific observations,44 measured-bright
observations in43 original contiguous groups. Two explicitly flagged copies
pad the final numerical group; they never enter estimates. Original input
IDs remain the RNG identities, with separate IDs assigned only to padding.

The single evaluation centre is the existing frozen value
`(0.009836059884222188, -0.00029219982162802493)`, chosen before looking at
these new mock results and independently of injected shear. It permits exact
reuse of the verified domain-matched selection normalization. This is a
fixed-centre first-pass diagnostic, not a converged Newton/BFGS chain.

All scientific rows first receive a baseline evaluation with Gaussian ranking
of131,072 candidates down to16,384 and the existing whole-prior tilted
complement. The choice K16,384 keeps the accepted full density-block size;
this baseline is explicitly different from the historical K1024 baseline.
For the44 rows with measured magnitude<20, substitute whole-prior exact
top16,384 plus the accepted exact-centre conditional-complement method.
Other rows remain exactly unchanged in the paired comparison. Evaluate all
three original seeds8701–8703 and every original draw rung, preserving the
four-observation/4096-atom density calls. No truth-property selection is used.

Preflight verifies that two original four-row groups give identical results
when evaluated together or separately. It also diagnoses the nearly seed-
constant small-K curvature residual by comparing direct K1024 candidate
sums with prefixes of the canonical16,384 evaluation on two old probes;
the corresponding K16,384 replay must pass within1e-7 in log space. This
diagnostic does not alter or repair any recorded result.

CPU initializer16732141 completed in16s:10 targeted proposal/reduction tests
passed, source hashes verified, measured cuts checked and padding recorded.
GPU preflight16732142 gates four persistent one-GPU workers16732143[0–3],
followed by CPU reducer16732144. Initial estimate is3–4hours on four V100s,
with6hour limits; runtime will be revised from progress. Status-only audit
16732145 is scheduled in about two hours. The reducer reports every seed,
draw count, baseline/repair pair and numerical-padding exclusion. It rejects
non-positive summed information rather than forcing a shear estimate.

Passing the old numerical pilot does not settle integration for every new
observation, intermediate-magnitude rows, finite-prior resolution, or model
misspecification. None of these first-pass results can by itself establish
robust0.3% calibration.

### Fresh-parent preflight and small-batch numerical effect

Preflight16732142 completed exit0 in3m21s. The two fresh four-row groups
replayed bit for bit, and both old-probe K16,384 candidate sums matched the
saved canonical sums within the required1e-7 log tolerance. The four
dependent workers16732143[0–3] then started on th-cl-nv01, one V100 each.

The direct K1024 diagnostic confirms batch-sensitive density numerics,
separate from Monte Carlo tail noise. It evaluates the same first1024 atoms
either directly or as a prefix of the canonical16,384 evaluation. For
actual5359 the candidate-only information difference is
`[[36.5221,-3.9134],[-3.9134,37.3339]]`; for mock3192 it is
`[[-41.3445,2.0822],[2.0822,-35.2963]]`. The underlying log-integral
differences are only about2e-5, amplified by the h0.001 second differences.
These deterministic shifts account for much of the seed-stable K1024
residual in the exact-tail pilot. The precise internal operation responsible
has not been isolated. Setting a maximum atom_chunk4096 alone does not make
a1024-candidate density call use a full4096-atom block.

This supports retaining K16,384 for the new diagnostic and explains why
accurate tail draws alone do not rescue K1024. It does not alter any old
reported result or prove that all remaining finite-prior/model errors vanish.

At3m52s worker elapsed time,40 of1283 baseline groups had completed all
three seeds, with no failures. Bright groups are distributed11/12/12/8
between the four workers. Startup checks confirmed one32GB V100 per worker;
the main run is progressing normally and the scheduled two-hour audit remains.

## Completed fresh-parent first-pass diagnostic

Workers16732143[0–3] completed exit0 in2h20m–3h12m23s; reducer16732144
completed exit0 in57s. All1283 baseline groups and43 bright groups finished,
with no failure receipts. The reducer verified all baseline/bright moment
hashes and bitwise equality of untouched nonbright rows. Completion review
verified the aggregate moments/protocol/input hashes, all worker and bright
group receipts, and independently reproduced the final Newton estimates from
the saved scientific-row moments. Both flagged padding rows remain excluded.
The status JSON is refreshed; no new compute job was submitted at this check.

The sample contains5,130 retained observations from one normal forward draw
of20,000 unselected-parent inputs, including44 measured-mag<20 observations.
Injected shear is(0.02,0). The following are **one fixed-centre Newton
proposals**, not converged shear estimates. Errors are the row-level sandwich
standard errors defined by the frozen reduction; these seeds reuse the same
observations and are not independent scientific realizations.

| Seed | Baseline g1 at16,384 draws | Bright-repaired g1 at16,384 draws | Repaired g2 | Repaired g1 movement,8192 to16,384 |
| --- | ---: | ---: | ---: | ---: |
|8701|0.018364±0.009940|0.016889±0.007046|−0.002062±0.002492|+0.000340|
|8702|0.017715±0.008945|0.016785±0.006936|−0.002061±0.002473|+0.000421|
|8703|0.017767±0.009038|0.017216±0.007600|−0.002206±0.002608|+0.000165|

All36 baseline/repaired seed/rung combinations have positive summed
information, but none meets the recorded small-step diagnostic. All six draw
rungs remain in result.json. The final repaired proposals are statistically
compatible with injected shear under these broad conditional error estimates;
this does not establish unbiasedness. The remaining8k→16k movements correspond
to0.83–2.10% of injected g1, exceeding the0.3% target scale of0.000060. The
16k seed range is0.000431. Neither numerical convergence at the target scale
nor0.3% scientific calibration has been demonstrated.

A small saved-array reduction in completion_audit.json attributes the Newton
step change exactly: with x=I^-1 S, each row contributes
I_b^-1[delta score_i−delta information_i*x_a] to x_b−x_a. The row sum replays
the observed changes within1e-14. The44 repaired bright rows contribute only
+2.76e-7,−2.36e-7,+2.80e-7 to the net g1 movement for seeds8701–8703.
The remaining movement therefore comes from measured-mag>=20 rows, mainly
20–24. Examples of influential numerical movements occur at measured
magnitudes20.46,22.50,22.57 and23.27; no observations are removed or reweighted
using this post hoc diagnostic.

Separately, the bright rows contribute88.8–89.3% of the squared g1 influence
entering the sandwich variance. Small aggregate draw-count drift in those rows
does not prove their finite-prior likelihood or local quadratic approximation
is accurate. Next numerical work should probe the intermediate-magnitude
rows, while keeping finite-prior resolution and optimizer convergence as
separate unresolved issues. No prior, model, threshold or production default
is promoted by this result.

## Intermediate-magnitude numerical pilot

The owner authorized continuing, with an explicit request to avoid
overengineering. The next experiment is one small comparison using the
existing algorithms: baseline Gaussian K16,384 with its original tilted
tail, versus exact whole-prior top16,384 with the already implemented
conditional exact-centre tail. Both use the same domain-corrected expanded
prior, models, normalization and frozen centre. No new estimator or tuning
parameter is introduced.

`scripts/probe_v39_intermediate.py` is frozen in
`inference_v39_intermediate_pilot_20260927_v1`. Four focal observations are
selected before new evaluations:

- In each measured-magnitude bin20–22 and22–24, choose the fresh-parent mock
  with the largest absolute contribution to8k→16k g1 Newton-step movement
  over all three existing seeds. These are explicitly post hoc stress probes.
- From the original actual20k, choose one uniform random observation in each
  of those bins, with fixed control seed2026092721 and no score/truth selection.

Keep each focal observation's original contiguous four-row density group.
Report all twelve incidental companions separately; they cannot silently
replace a failing focal row. Fresh-mock object IDs and actual original row
IDs preserve their random streams. No sample is generated or cut further.
The new script reuses the frozen parent evaluator, exact-tail helper,
likelihood loader and finite-difference reduction.

For each group, sum all28,012,032 aligned prior atoms at all nine existing
shear stencil points, with h=.001 and atom_chunk4096. In the same scan retain
centre contributions and top16,384 atoms, avoiding a separate centre scan.
Original four-row shapes and full candidate density blocks remain fixed.
The two fresh baseline evaluations must replay all saved arrays exactly;
selected exact-centre densities must replay within1e-7 log units. All three
seeds8701–8703 and six draw rungs are retained. The unchanged entrywise gate
is abs(score error)<=.1+.001*abs(reference), and
abs(information error)<=10+.001*abs(reference), for both8k/16k draws and
every seed/focal row. Companion outcomes are reported separately, including
failures. Exhaustive sums also retain distinct-atom ESS and maximum atom mass
to keep finite-prior concentration separate from Monte Carlo error.

CPU preparation16733294 runs the ten existing helper/reduction tests and
verifies frozen inputs before four one-GPU V100 workers16733295[0–3]. CPU
reducer16733296 follows successful workers; status-only audit16733297 is
scheduled about two hours after submission. Four GPUs total, three-hour
worker limits. This numerical gate is not a shear-accuracy acceptance test.
The original actual20k and matched16,314 full-sample failures remain open.

Preparation16733294 completed exit0 in22s; all ten helper/reduction tests
passed. The frozen selections are fresh-mock row4826 (measured mag20.45964),
fresh-mock row3602 (22.57042), actual row11173 (21.1134), and actual row2510
(22.3313). The four workers started on th-cl-nv01, each with one32GB V100.
Source/frozen-driver hashes match, and scheduled resource receipts confirm
the four-GPU allocation. Both fresh baseline groups then replayed all saved
arrays bit for bit across all three seeds; the two actual baselines also
completed. All four exhaustive scans reached1,064,960 of28,012,032 atoms
in50.4–50.8s of scan time, with all nine nodes evaluated per block and no
failure receipts. Initial steady throughput suggests about30–40minutes
remaining including the tail comparisons. Centre candidate replay and the
final numerical gate remain pending. The normal-progress jobs are left
running with their dependent reducer and scheduled status audit.

## Completed intermediate-magnitude pilot

Completion reviewed on September28 local time. All four workers16733295
completed exit0 in26m16s–26m20s; reducer16733296 completed in23s. The full
scans cover all28,012,032 rows and all nine stencil points. Worker/protocol
receipts, centre-density replay and all small output hashes were rechecked;
the scheduled reducer also verified every retained centre-array hash.

| Checks (each combines score and information tolerances) | Baseline passed | Repaired passed |
| --- | ---: | ---: |
| Four focal rows × three seeds × two draw rungs |0/24|24/24|
| Twelve incidental companions × three seeds × two draw rungs |24/72|72/72|
| All sixteen rows |24/96|96/96|

Every predeclared focal check passes for the repaired method; none passes
for the baseline. Across the four focal rows, maximum absolute score error
falls from110.5340 to0.05604 and maximum information-entry error from12390.74
to0.84434. Including all twelve companions, repaired maxima are0.08432 and
1.29420. Tolerances and seeds are unchanged. The existing baseline therefore
has consequential numerical errors in both mock stress probes and both
random actual controls, extending the earlier bright-row finding.

| Focal row | Exact information eigenvalues | Distinct-atom ESS | Largest atom mass |
| --- | --- | ---: | ---: |
| Fresh mock4826, mag20.46 |−52736.84, +112237.28|2.480|62.284%|
| Fresh mock3602, mag22.57 |−2111.32, +304.69|10.251|27.361%|
| Actual11173, mag21.11 |−2911.54, +4751.49|7.946|29.185%|
| Actual2510, mag22.33 |−18.13, +5.70|1212.460|1.890%|

Information is minus the local log-likelihood Hessian. All four exact
per-observation matrices are indefinite at this centre. Passing the gate
means reproducing that curvature, not forcing it positive. Individual
indefiniteness neither proves that the full-sample matrix is indefinite nor
establishes bias; it can occur in a mixture likelihood. The first three
posteriors remain sparse, motivating a separate finite-prior-resolution
check, but sparsity has not been established as the cause of the old failure.

The exact top16,384 candidates cover at least99.764% across the stencil for
the first three probes, but only28.590% for actual2510. The complementary
importance estimate is therefore essential; this result does not justify
discarding the tail or treating high candidate coverage as a general rule.

This completes the small integration validation requested before returning
to the original failing cohorts. The old actual20k negative information and
matched16,314 mock proposal g1=0.1473 have not yet been recomputed in full
with the repair. The current results are numerical evidence on sixteen
observations, not recovered full-sample shear or0.3% calibration. No new
algorithm, model, weight correction or production default is introduced.

## Original-cohort recovery comparison

**Status update September28:** stopped at the owner's request for smaller
test jobs, preserving293 baseline chunks and all inputs/logs. No full-cohort
repair result exists. The [short prefix diagnostic](#short-original-cohort-diagnostic)
below reuses completed baseline work; the full run will not resume automatically.

On September28 the owner authorized the next comparison. The original
actual20k and the original matched16,314 scientific mocks are retained,
including the mock's original3,686 flagged numerical padding rows in their
original four-observation groups. Padding never enters estimates. No new
observations, measured selection, truth cuts or recentering are introduced.
Both targets retain the additional bright catalogue: the two prior arms are
the originally expanded weights and the previously verified domain-corrected
weights, each with its own existing selection normalization.

`scripts/run_v39_cohort_recovery.py`, frozen in
`inference_v39_original_cohorts_recovery_20260928_v2`, crosses these two
priors with baseline and measured-bright-repaired integration. Baselines use
K16,384 and the same original expanded-prior Gaussian/tilted proposal for
both target priors. Common candidate sets, random streams and importance
probabilities make the comparison sensitive to the target weight and
normalization change without an additional proposal change. The original
proposal has positive support for all active corrected-prior atoms. This
corrected baseline differs explicitly from the earlier corrected-proposal
baseline; neither replaces a frozen result.

Repair is limited, initially, to **all135 actual and134 mock scientific
rows with measured magnitude<20**, in266 original groups. This targets the
bright population that dominated the original failure while keeping the
first full-cohort comparison bounded. All fainter measurements remain in
the sample with their baseline moments; their numerical errors are still
unresolved. Report every seed and draw rung, including remaining drift. This
is partial integration repair, not full-sample numerical acceptance.

For each repair group, one full expanded-prior centre scan retains all
atom contributions. Corrected centre contributions follow by multiplying
by corrected/expanded prior weights; no new model evaluation is required.
Keep distinct exact top16,384 sets and the existing conditional exact-tail
proposal for each target prior. The direct target-density replay must match
each selected set within1e-7 log units. The two existing normalization
tables remain unchanged. One original centre array is retained per group;
the corrected array is reconstructed when needed. This avoids duplicate
expensive scans, without changing the target integral.

Preflight bridges to the original failing calculation: replay historical
expanded K1024 moments bit for bit for actual1958 and mock4460; replay
corrected K16,384 selected-density sums across all nine stencil points;
verify the prior-weight-ratio identity against direct density evaluations,
including8,192 zero-weight added atoms. Exercise K16,384 evaluation of both
targets using the same proposal. No tolerance is widened to enable the run.
Production defaults and all model artifacts remain unchanged.

CPU preparation16733514 runs the ten existing helper/reduction tests and
checks source catalogues. GPU preflight16733515 gates four persistent
one-V100 workers16733516[0–3]; CPU reducer16733517 follows their success.
Worker allocations are bounded at36hours. Initial total estimate is20–24
hours on four GPUs, based on the completed baseline throughput and266
centre scans; refine it from actual progress. Status-only audit16733518 is
scheduled about two hours after submission. The earlier completed pilot's
pending status audit is CPU-only, so it does not consume the GPU allowance.

The reducer reports actual_all20k, actual_same_ids and mock samples for
all three seeds8701–8703 and six draw counts, with the four prior/integration
arms, summed scores/information, Newton proposals when defined, sandwich
errors, paired prior/repair changes, and8k→16k comparisons. It asserts that
nonbright moments are untouched by repair. It does not force positive
information or run an optimizer. Matched mocks remain a doubly selected
diagnostic, and no0.3% acceptance follows from this comparison alone.

The initial v1 preparation16733505 passed ten tests and input checks in22s.
Preflight16733506 was stopped during startup, and its pending workers16733507,
reducer16733508 and status16733509 were cancelled before scientific workers
allocated. Review found that replaying historical primary/adaptive arrays
requires its original min_ess32/max_fraction.5 settings, whereas the newer
parent evaluator defaults to256/.25. The full draw ladders are the scientific
comparison, but that does not make the primary arrays interchangeable.
V2 calls the frozen historical evaluator directly for this replay, retaining
all existing K16,384 comparison settings. V1 source/protocol and its explicit
superseded receipt are preserved; no completed scientific output is replaced.

V2 preparation16733514 completed in12s with ten tests passed and unchanged
scientific counts. Preflight16733515 passed on a32GB V100: actual1958 and
mock4460 reproduce every saved historical array bit for bit. Maximum
corrected canonical log-sum discrepancies are1.05e-13 and1.40e-13. Direct
prior-ratio comparisons, including zeroed added atoms, pass at all checked
views; both targets run with the common proposal. Maximum allocated GPU
memory was18.07GB. The four dependent workers are permitted to start.

Preflight completed exit0 in5m04s. All four workers16733516[0–3] then
started on th-cl-nv01, each exposing one32GB V100. Startup was followed
through completion of the first original four-row group on every worker,
covering both target priors and all three seeds, without failures. Inputs
are partitioned into314 baseline chunks (78–79 per worker) and266 repair
groups (66–67 per worker). The active/frozen driver hashes match. Leave the
normally progressing long run with its dependent reducer and two-hour
status audit; the20–24hour estimate remains provisional until checkpoints
give sustained throughput.

## Short original-cohort diagnostic

The owner requested smaller tests on September28. Cancelled full workers
16733516 and their reducer/status jobs while retaining157 mock and136
actual baseline chunks. No bright-group repair had completed. This avoids
waiting for266 scans over28,012,032 prior atoms each. Completed checkpoints
remain available, but the full run is not scheduled to resume.

The new frozen root is `inference_v39_cohort1024_recovery_20260928_v1`.
Use the **first1024 original identity slots** in each cohort, chosen without
looking at their inference results. Keep numerical mock padding in its
original density groups and exclude it from statistics. Report the actual
scientific mock count and the actual measurements for those same IDs.
This prefix is a small diagnostic, not a new independent validation sample.
Preparation counted1024 actual measurements and836 scientific mocks, with
188 numerical padding rows excluded from the mock summary. The matched-ID
actual summary therefore also uses836 measurements.

Reuse all sixteen completed baseline chunks after checking their hashes,
row ranges and scientific protocol settings. Keep both expanded/domain-
corrected priors, all three seeds and six draw rungs, the unchanged centre
and normalization, K16,384 and original four-observation density batches.
Only four scientific measured-mag<20 rows occur in each prefix: original
IDs32,38,179,582. Thus eight original groups need centre scans, two per GPU.
No observations are discarded for having unstable scores or information.
All fainter observations retain their baseline moments.

The existing cohort driver gained prefix selection and verified baseline
reuse; no separate inference algorithm or new launcher framework was added.
Preparation16735074 runs ten existing helper/reduction tests and verifies
the copied arrays. The original successful GPU preflight is inherited
explicitly. Each repair still verifies its selected centre densities against
the full scan. Four one-V100 workers16735075[0–3] are bounded at2hours;
reducer16735076 follows success. Estimated runtime30–45minutes plus queue
time, to be checked against the new scans. Status audit16735077 is CPU-only.
Preparation passed all ten tests and input/cache checks in25s. All four
workers reached their first centre scan without failure after loading and
compilation; their baseline chunks were reused, not recalculated.

Compare the summed information eigenvalues, scores and any valid Newton
proposals across both priors and baseline/bright repair, including seed and
8k→16k changes. Do not force positive curvature or describe a one-step
proposal as a converged estimate. This sample cannot establish0.3% bias;
its four bright objects per arm need not reproduce the full20k failure.
The matched mocks' double selection and fainter numerical errors remain.

## Completed short cohort diagnostic

**Comparison boundary:** the earlier full16,314 matched-mock expanded-prior
run had positive summed information at16k draws, but an unstable Newton
proposal g1=0.1473 against injected0.02; the full20k actual-data run remained
indefinite. See the [original expanded-prior result](V39_MATCHED_INPUT_MOCK.md#completed-expanded-prior-result-on-september27).
The current1024/836 prefix and changed numerical integration are a different
test. Its negative information under both weight choices does not establish
that reweighting made the original full-mock information negative. Both
current arms retain the bright extension; the corrected arm gives zero mass
to out-of-generator-domain added atoms and preserves the total bright mass.
The controlled full-cohort reweighting/repair comparison was stopped before
completion, so that full-cohort outcome remains unknown.

Workers16735075[0–3] finished successfully in33m30s–33m47s and reducer
16735076 in20s. All eight repaired groups passed their selected-centre
density replay checks. The completion review verified the protocol and
combined-moment hashes, all four worker receipts and all small repair-array
hashes, and confirmed unchanged fainter moments. It did not rehash the
large centre arrays. Full evidence is in result.json and completion_audit.json
under `inference_v39_cohort1024_recovery_20260928_v1`. No GPU jobs remain;
the original full-cohort continuation remains stopped.

At16,384 draws the domain-corrected prior gives these minimum eigenvalues
of summed information (each column is one sampling seed):

| Sample and integration | 8701 | 8702 | 8703 |
|---|---:|---:|---:|
| Actual1024, baseline | -113822 | -108529 | -60597 |
| Actual1024, bright repaired | -315697 | -359242 | -296517 |
| Mock836, baseline | -16530 | +31949 | +60986 |
| Mock836, bright repaired | -42441 | -38947 | -70409 |

The expanded-prior repaired arms also have negative minimum eigenvalues in
all three seeds, as does the actual same836-ID comparison under both priors.
These are fixed-centre information diagnostics, not bias measurements or
converged shear estimates. No positive-definite Newton estimate is accepted
for any repaired16k arm. Some lower rungs are positive, so the sign statement
is explicitly restricted to the final16k rung. Reweighting alone has not
resolved the failure in this subset.

Doubling draws8k→16k changes each corrected repaired bright score component
by at most0.00901 and information entry by at most8.85. This supports bright
draw stability here, but is not a new exhaustive nine-node accuracy proof.
The fainter terms retain much larger draw and seed variation.

For each seed, project every object's information onto the minimum-eigenvalue
direction of the corrected repaired total. In actual measurements, bright
objects contribute about-246k and fainter objects add-50k to-113k; bright
row32 contributes about-249k by itself. In mocks, the four bright objects
have a positive net contribution(+34k to+35k), while fainter rows contribute
-74k to-104k. Mock rows151,734,853 are prominent negative contributors.
Directions differ by seed and sample; these decompositions explain their
respective totals and are not common-direction population comparisons.

The bounded next diagnostic is to inspect exact local likelihood behavior
for actual bright row32 and the dominant fainter mock rows, separating
remaining integration error from local curvature of the finite-prior
likelihood. No new run was launched as part of this status review. The small
cohort does not resolve the original full20k outcome or validate0.3% bias.

## Paired exact curvature probes

The owner requested continuation on September28. The next bounded diagnostic
uses actual row32 and mock rows151,734,853, retaining each original group of
four observations. These are explicitly post hoc stress probes selected
from the completed1024/836 subset's negative-information contributions.
The four focal objects and incidental companions cannot provide a
representative acceptance claim.

Source: `scripts/probe_v39_cohort_curvature.py`, frozen in
`inference_v39_cohort_curvature_20260928_v1`. Each worker evaluates all
28,012,032 atoms at the existing nine h=.001 stencil points around
(0.009836059884222188,-0.00029219982162802493). From each expanded-prior
log contribution, add log(corrected weight/expanded weight) to obtain the
domain-corrected contribution. Direct-density checks at every stencil node
include8192 positive-weight and8192 zeroed atoms. Both targets retain their
existing separate64-QMC selection normalizations. Four-observation and
4096-atom density batches remain fixed; full scans use contiguous16384-atom
blocks. Models, population cuts, seeds and likelihood are unchanged.

Retain exact numerator/selected log sums, second moments and maximum atom
contributions for all nodes and both priors. Replay the old full centre
array for actual rows32–35; save centre arrays for the three mock groups.
Compare exhaustive score/information with the saved subset baseline and
bright-repaired moments at8k and16k over all three seeds. Use the existing
score atol0.1, information atol10 and relative tolerance0.001; failed checks
are reported without changing the tolerance.

The reducer also replaces only the scientific rows in the four probed
groups in saved bright-repaired subset totals. It verifies all untouched
moments remain bitwise identical and reproduces the source aggregate
summaries before replacement. Actual1024, actual same836 IDs and mock836
are reported for both priors, three seeds and8k/16k. Padding remains in the
numerical groups but never enters totals. This isolates how much these
specific contributions explain; all unprobed rows remain approximate.

Preparation16735500 runs ten existing helper/reduction tests and verifies
source receipts and the retained actual centre array. Four one-V100
workers16735501[0–3] are capped at2hours, followed by reducer16735502.
No other study GPUs were active. Initial estimate30–45minutes, to be
refined from startup and scan progress. Status-only audit16735503 runs
about two hours later. The original full-cohort comparison remains stopped;
there is no automatic further experiment or model promotion.

Preparation completed in24s with ten tests passed. The groups contain15
scientific rows and one padding row(mock732);36 source aggregate comparisons
replay exactly before replacement. All four32GB V100 workers passed direct
weight-ratio/zero-support checks at all nine stencil points and advanced
beyond1,064,960 atoms each without failure. First65 blocks take about51s,
indicating roughly22minutes for the full scan before final reductions and
receipts. Final results remain pending.

Exhaustiveness applies to the finite-prior numerator and the chosen stencil,
not an analytic population likelihood, step-size convergence, or a global
likelihood optimum. The remaining64-QMC normalization uncertainty, finite
prior resolution, fainter/unprobed integration errors and matched-mock
double selection still limit interpretation. This cannot validate0.3% bias
or answer the uncompleted full-cohort reweighting comparison by itself.

## Paired-probe replay recovery

The three mock workers completed in27m27s–27m45s. The actual worker failed
after10m41s on the retained-centre replay: one log contribution differed by
1.43686243e-7, just beyond the fixed1e-7 absolute tolerance. The combined
reducer did not run. Preserve all v1 outputs and failure receipts.

The completed exact mock arrays were hash-verified at the owner's status
check. Minimum per-object information eigenvalues are:

| Mock row | Expanded weights | Domain-corrected weights |
|---|---:|---:|
|151| -46145.90 | -46150.64 |
|734| -28283.90 | -28283.91 |
|853| -25551.01 | -28115.16 |

These local negative directions survive summing every prior atom, under both
weightings. The original sampling still has large errors, including16k
information-entry discrepancies up to18606. Corrected centre distinct-atom
ESS is6.44,5.76,8.11, with maximum atom masses36.1%,31.8%,30.5%. Neither
draw convergence nor exact local negative information alone diagnoses global
shear bias. This is a partial result for three post hoc mock probes; no
combined actual/mock or full-cohort result is available yet.

The saved actual centre array was evaluated at the original full-precision
centre; the new scientific stencil rounds node coordinates to14 decimals.
Their centre components differ by2.1875e-15 and1.9751e-15. The replay repair
in `inference_v39_cohort_curvature_20260928_v2` first tests both representations
on the atom interval[6291456,7356416), which contains the failure. It must
reproduce the rounded-input violation and pass the original-input replay at
the unchanged1e-7 tolerance. The rounding explanation remains a hypothesis
until this GPU check passes.

The scientific nine-node evaluations remain unchanged, including their
rounded centre; a separate unrounded-centre evaluation is used only to verify
the old retained array. Completed mock groups are reused with their original
receipts, after comparing the two protocols' scientific settings. Only the
failed actual group reruns. Preparation16738602 gates one-V10016738603_0
and reducer16738604; status-only audit16738605 follows in about two hours.
Cancelled the impossible old reducer16735502. No tolerances were relaxed,
and no failed or completed scientific output was overwritten.

The v2 GPU check confirmed the replay-input explanation: the rounded centre
reproduces one1.4368624334792912e-7 violation on the checked interval, while
the original unrounded centre matches all retained values exactly (maximum
error0). The prior-weight checks pass at all nine nodes. Worker16738603_0
has entered the exhaustive scan; the completed mock groups remain reused.
Preparation passed all ten tests in27s. Only the comparison against the
old centre array changes; scientific evaluations and tolerances are fixed.

## Completed paired exact curvature probes

Recovery worker16738603_0 completed in27m43s, followed by reducer16738604
in24s, both exit0. All four groups are complete in
`inference_v39_cohort_curvature_20260928_v2/result.json`, with three mock
groups reused from v1. The full actual-centre replay has maximum error0;
the rounding fix changed no scientific stencil sums or tolerances. All
direct prior-weight/zero-support checks pass. The reducer verified output
hashes, including large centre arrays, reproduced36 source aggregates,
and left unprobed rows bitwise unchanged. A separate light completion audit
verified pinned receipts and all exact-array hashes, then reproduced all32
reported per-object/prior moments from the saved nine-node sums.

Information means minus the finite-difference Hessian of selected log
likelihood at the fixed centre and h=.001. Exhaustive focal results:

| Observation | Minimum eigenvalue, expanded | Minimum eigenvalue, corrected | Corrected distinct-atom ESS | Largest corrected atom mass |
|---|---:|---:|---:|---:|
| Actual32, mag19.4751 | -248854.33 | -253958.42 | 4.57 | 45.54% |
| Mock151, mag21.0461 | -46145.90 | -46150.64 | 6.44 | 36.11% |
| Mock734, mag22.1887 | -28283.90 | -28283.91 | 5.76 | 31.79% |
| Mock853, mag20.1350 | -25551.01 | -28115.16 | 8.11 | 30.47% |

The actual focal row's repaired sampler passes all12 existing moment gates
(two priors, three seeds,8k/16k), while its baseline passes0/12. Maximum
corrected repaired score-component error is0.01383 and information-entry
error15.13; the gate includes its unchanged relative tolerance. Thus the
large negative direction is reproduced by both repaired sampling and
exhaustive sums. Each mock focal row is above the measured-mag20 repair
threshold, so its baseline and bright-repaired moments are identical.
Those three focal mocks fail all36 checks per arm, but the exhaustive results
still have negative directions. Missing sampled atoms are not the sole cause.

Replacing only the15 scientific rows in the four groups leaves all36
partial-cohort summaries indefinite, for both weightings, all three seeds,
8k/16k, and actual1024/actual-same836/mock836. At16k:

| Cohort | Expanded minimum eigenvalue range | Corrected minimum eigenvalue range |
|---|---:|---:|
| Actual1024 | -351445 to -307678 | -359262 to -296561 |
| Mock836 | -69260 to -51913 | -64507 to -45578 |

No valid Newton estimate is produced. These are partial replacements; the
unprobed rows still use sampled moments. Reweighting alone has not repaired
this diagnostic. The earlier full16314-mock expanded-prior positive
information remains a separate result, with an unstable one-step estimate;
the controlled full-cohort reweighted comparison remains unfinished.

The high concentration suggests finite-prior resolution deserves attention,
but it does not prove the prior population is wrong or that a global optimum
is biased. Negative per-object information can occur away from a likelihood
maximum. Exhaustive atom summation also does not establish convergence in
shear spacing or remove the existing64-QMC normalization approximation.
A bounded next diagnostic would inspect these same objects' local likelihood
profiles and smaller shear spacings before changing the prior or launching
another large run. No new experiment was submitted at this status check.

## Reflection on the failure mechanism

On September28 the owner requested a pause to interpret the pattern across
same-prior mocks, actual measurements, matched-input mocks, and the bright
extension. No new compute run is implied by this reflection.

The earlier20k same-prior mock is an internal finite-model closure test:
generation draws weighted atoms from the exact catalogue later integrated
by inference. Its generating atom is therefore present. For narrow conditional
likelihoods, this can matter even if a separate catalogue has the same parent
distribution and excellent marginal histograms. The bright-only BFGS chain
settled near the injection, but failed draw convergence; it also carried
fainter scores linearly. It did not validate inference for independent
continuous-parent galaxies or establish0.3% bias.

There are two different resolutions. Increasing the number of proposal draws
approximates the fixed finite-catalogue sum more accurately. Increasing valid
independent prior support changes the finite sum's approximation to the parent
integral. Exhaustively summing a sparse catalogue solves only the first.
Conversely, adding millions of atoms does not ensure the proposal finds the
few that dominate each narrow likelihood. The paired exhaustive probes prove
large integration errors; their small distinct-atom ESS makes the second
resolution a concern, without proving it causes global bias.

Observed information is not required to be positive at an arbitrary shear.
For positive atom contributions a_j(g), posterior fractions r_j=a_j/sum(a),
scores s_j=gradient(log a_j), and atom information I_j=-Hessian(log a_j),
the selected per-object information is

`I = E_r[I_j] - Cov_r(s_j) + Hessian(log P_selected(g))`.

This is the vector version of [the existing mixture derivation](MATH.md#5-derivation-of-inferencemd-58).
Competing atoms with different shear scores subtract information through
the covariance term. As shear changes, posterior mass can switch between
atoms, producing locally convex portions of a valid mixture likelihood.
For example, two equally weighted Gaussian peaks at g=+a and-a with width
sigma have information1/sigma^2-a^2/sigma^4 at their midpoint, negative
when a>sigma. This is an explanatory example, not a fitted model of our data.
Expected Fisher positivity under the model at the generating parameter does
not require every observed Hessian, or a sum evaluated elsewhere, to be
positive. Our actual/matched probes are centred at g1=0.009836, not a located
stationary point. Their failed initial observed-information step prevented
any of the intended BFGS recentering iterations from running.

The old expanded-prior mock's positive information was itself based on an
unconverged atom integral. Its large Newton proposals do not locate a
likelihood maximum. A low-draw estimate can miss competing atoms and have
apparently benign score/curvature; additional draws may expose that omission.
The actual full cohort's information changed back from positive at8k to
negative at16k. These patterns cannot justify choosing a favorable draw rung.
The latest exhaustive probes establish local non-concavity of the chosen
finite-prior stencil, not convergence in stencil spacing or the global shear.

The matched-input mock is also conditional on the original image selection
and undergoes a second model selection. As derived
[above](#why-the-matched-input-mocks-have-a-different-target-distribution),
it is not generated from the ordinary unselected-parent likelihood being
fitted. Its failure establishes that image/model measurement mismatch is
not necessary for this failure, but does not isolate the prior, sampler,
or optimization as its unique cause.

The clean unselected-parent5130 result must remain in the comparison:
all36 baseline/repaired seed/rung combinations had positive information.
Its repaired one-step g1 values were0.0168–0.0172 with conditional errors
about0.007, and it failed numerical convergence. It is neither a success
claim nor evidence that every off-prior mock fails. Similarly, the original
prior's close population/crowding histogram agreement argues against an
already established broad population mismatch. The bright-domain correction
is justified, but does not settle finite support, full joint population
agreement, or optimizer behavior.

A useful discriminator would hold models, parent sampler, selection and
integration accuracy fixed: generate a small selected mock from prior A,
then infer using A and an independently realized prior B from the same
unselected parent, with comparable stratification and population weights.
Evaluate stable likelihood profiles rather than judging a single Newton
proposal or curvature sign. Sensitivity to A/B and to valid prior density
would test finite-realization dependence; a correct profile maximum despite
negative initial information would instead demonstrate a local solver issue.
Reuse the existing fresh-parent mocks as the independent-input control.
Do not infer new population weights from the selected truth histogram.
This experiment is proposed only; the current request is reflection.

## Independent prior profile comparison

The owner subsequently authorized the comparison described in the reflection,
explicitly correcting a proposed atom-thinning diversion. No thinning job was
submitted. The frozen experiment is
`inference_v39_independent_profiles_20260928_v4`, driven by
`scripts/run_v39_independent_profiles.py`.
The first three attempts and their diagnostics are preserved. All scientific
settings,87 observation-group hashes and normalizer input hashes remain equal
to v2; see `repair_identity_audit.json` in the current root. The preflights and
all87 groups completed. The coarse comparison is summarized under
[completed independent profiles](#completed-independent-profiles); no accepted
shear estimate is available.

Use two already generated, disjoint scene sets from the original F8-ready
prior. A covers scenes20000–20099 with10,116,902 atoms; B covers20100–20199
with10,114,319 atoms. Both use the same empirical parent source and original
inverse-inclusion weight rule. Their original total masses are0.5000549108
and0.4999450892, respectively; divide by these separately. This is independence
of generated scene realizations, not independence from the common empirical
source galaxy catalogue. It is not arbitrary atom thinning, a newly fitted
population, or the bright extension. Each prior has roughly half the old
support density, which limits extrapolation back to the complete20m prior.

Take the first128 observations in the existing original-prior20k mock whose
generating atom lies in A. Conditioning those retained draws on membership
in A produces the selected model distribution for A; no fresh forward noise
or extra selection is introduced. The control takes the first128 measured
outputs from the fresh unselected-parent5130 mock. It was selected once,
and its generating inputs were never inserted into the inference prior.
No truth-property cuts are applied. Preserve original four-row likelihood
groups and mark companions non-scientific. This produces87 numerical groups,
but128 scientific observations per cohort. There are three measured-bright
objects in the A cohort and none in the control prefix; the control therefore
does not characterize bright behavior in this small realization.

Evaluate both cohorts under A and B. Freeze a two-dimensional7x7 grid at
offsets-0.03 through+0.03 in0.01 steps around the full original20k mock's
measured mean (0.01374048162339,-0.00264758006175). Do not choose the centre
or grid from injected shear. At the centre, scan the whole original catalogue
for each original observation group and keep the exact top16384 atoms per
prior. Reuse the existing exact-centre conditional-complement method with
delta0.1, two fixed proposal seeds and nested4096/8192/16384 draws. Retain all
variants and exact-centre sums. Four-row and4096-atom density batches stay fixed.

Compute separate selection normalizations for A and B at every grid point.
Each has two independent weighted-prior draws, nested32768/65536 atom counts,
and32/64 common-random QMC flow draws per atom. Report both sources of numerical
sensitivity; the normalizer is estimated, not exhaustive. The initial profile
screen uses0.05 absolute centred-log-likelihood difference in the recorded
near-maximum region, retaining the full arrays for independent review and
checking alternate modes before any acceptance. Grid-boundary modes cannot
be accepted, and any coarse interior maximum still needs refinement.

Before the broad calculation, the first original group from each cohort is
checked at the centre and g1±0.01 against exhaustive A/B numerator sums.
Every row, both priors, both seeds, and8k/16k must agree to0.001 in centred
log numerator. This is an integration check; passing it does not prove
accuracy at every farther grid point. Failures halt the broad dependency chain.

### Replay and execution history

Preparation16742796 passed ten helper/reduction tests and all input, prior
mass and scene-disjointness checks in33s. One-V100 preflight16742809 gates
four persistent one-V100 workers16742810[0–3], followed by16742811. The v1
attempt stopped on a module-writer alias before producing any science output;
its logs are preserved. V3.9 identities, all original results and production
defaults remain unchanged. This is a small mechanism diagnostic, not a0.3%
calibration experiment. Results and profile refinement remain pending.

The v2 preflight subsequently failed its per-atom centre replay after the
full-centre scan:6/65,536 reordered candidate terms differed by more than
1e-7, with maximum3.69084e-6. This is numerically small but unexplained;
the replay tolerance was not relaxed and broad workers did not start.
Frozen v3 retains all scientific settings, adds saved failure arrays and
fixed-group response reuse, and first runs a bounded order/reuse diagnostic.
The reuse checks exact coefficients against the original implementation;
the density gate remains1e-7. CPU preparation16742917 and GPU diagnostic
16742918 are the current chain. Prior conclusions remain pending.

V3's bounded probe completed with a numerical failure: full16k atom blocks
were bitwise invariant under reordering, whereas the final13,365-atom block
had5,777 differing terms confined to its fourth observation, max0.1183698.
Coefficients matched exactly throughout. That observation crosses the final
short density batch. This is a batching issue, not evidence for a population
change, and does not yet explain v2's six small candidate discrepancies.
V4 pads the last atom block with duplicates excluded from the scientific
sums/rankings and first checks fixed permutations, mixed subsets and duplicate
queries. Preparation16742975 gates bounded replay16742976. No density batch
size, observation grouping, model, prior, cohort, seed or tolerance was relaxed.

V4 preparation passed11 tests and all receipts. All ten bounded replay cases
passed with zero coefficient and density differences. Two one-V100
preflights16743015[0–1] now gate the CPU check16743016, then four one-V100
workers16743017[0–3] and reducer16743018. The original v2 candidate discrepancy
still has to pass the full-centre replay; bounded probes do not establish that
it is fixed. Actual A/B profile results remain pending.

## Completed independent profiles

All87 groups and four normalizer workers completed on September28; audited
on September29. The slowest GPU worker took3h51m42s. All coefficient and
candidate-density replay checks passed. Both three-node exhaustive preflights
passed the0.001 threshold, with maximum centred-log-numerator errors0.000390
and0.000361. CPU audit16746010 verified all87 output hashes and independently
reproduced all four centred likelihood profiles within1e-10. Artifacts are
under `inference_v39_independent_profiles_20260928_v4/audit/`, including
`audit.json`, per-observation arrays and `profiles.png`/`profiles.pdf`.

| Cohort (128 each) | Prior | Coarse grid peak (g1,g2) | Numerator variation | Normalizer variation |
|---|---|---|---:|---:|
| A-generated | A | (0.02374048, -0.00264758) | 0.00933, pass | 0.01191, pass |
| A-generated | B | (0.02374048, -0.00264758) | 0.00745, pass | 0.01790, pass |
| Fresh-parent | A | (-0.00625952, 0.01735242) | 0.12726, **fail** | 0.02589, pass |
| Fresh-parent | B | (0.02374048, -0.00264758) | 0.02760, pass | 0.02436, pass |

Variations are absolute differences in centred total log likelihood over
the union of the recorded near-mode regions; the threshold is0.05 for each
source of numerical sensitivity. Grid spacing is0.01. None of these peaks
is a refined estimate or evidence for0.3% calibration. For the A-generated
cohort, some remote low-likelihood grid nodes remain unstable even though
the declared near-mode gate passes.

The same A-generated coarse peak survives changing scene realization, but
this does **not** cleanly remove generating-galaxy intrinsic support. CPU
overlap audit16746043 found19/128 A inputs have exact matches in B for all
five intrinsic features (23 B atoms in total), including all three measured
mag<20 observations: rows116,208,245 have3,1,2 such matches. None of the128
has an exact match in all eight F8 features; crowding differs. Complete
response graphs were not compared. This supports the existing qualification
that A/B are disjoint scene realizations sharing an empirical source, not
fully independent intrinsic-galaxy catalogues.

This overlap is relevant to the likelihood, not merely catalogue bookkeeping.
Saved exact centre sums and candidate terms give the following posterior
fractions at the fixed starting point(0.01374048,-0.00264758):

| A-mock row | Generating atom's posterior mass in A | Intrinsic-matched atoms' combined posterior mass in B |
|---|---:|---:|
| 116 | 66.71% | 68.63% |
| 208 | 99.23% | 81.32% |
| 245 | 38.38% | 99.973% |

Every listed matched atom is present among the saved exact top candidates,
so these fractions use the exhaustive centre denominator, not a sampled
normalizer. They apply only at this centre. This substantially limits how much
the A/B agreement can say about inference without generating-galaxy intrinsic
support. See `audit/bright_intrinsic_support.json`; the repeat CPU overlap
audit16746067 saves actual matched atom IDs.

Fresh-parent row47, measured magnitude20.76839, causes most of the remaining
draw instability: max8k-to16k change0.18904 in its A log numerator. Other
rows with small numerical errors also contribute to A/B differences. A prefers
its own grid peak over B's by3.2303 log-likelihood units; B prefers its own
over A's by5.5145. These are provisional grid-point comparisons, not calibrated
significances. Re-evaluating the original four-row group66, rows44–47,
exhaustively at every grid node will remove the main integration ambiguity
without dropping observations or changing prior masses. Four workers16746033
and reducer16746034 are running under
`inference_v39_independent_exact_group_20260929_v1`.

The exact group repair leaves the other124 rows and all normalizers unchanged.
Every worker first replays the saved exact centre to1e-7. Full-grid profile
variation must remain within the existing0.05 threshold. Finer grids and
stronger independence checks remain necessary before concluding how much
same-catalogue closure depends on finite intrinsic support.

## Completed exact group profile

Workers16746033 completed in22m35–39s and reducer16746034 completed in41s.
All49 grid nodes are now exhaustive for fresh-parent group66 (rows44–47)
under A and B. The other124 rows and both selection normalizers are unchanged.
Every worker replayed the source exact centre to7.8613e-11, within1e-7.
An independent saved-array audit verified ten source hashes, all worker
hashes, node coverage and reconstruction of the aggregate within1e-10.
Results and `completion_audit.json` are in
`inference_v39_independent_exact_group_20260929_v1`.

| Prior | Coarse peak (g1,g2) | Seed/draw variation | Own peak preferred over other prior's peak |
|---|---|---:|---:|
| A | (-0.00625952, 0.01735242) | 0.04542, pass | 3.2310 log units |
| B | (0.02374048, -0.00264758) | 0.02861, pass | 5.5163 log units |

Both peak coordinates are unchanged. The exact correction at either competing
peak is at most0.00188 log units, so this repaired group's sampling error
does not explain the observed A/B mode separation on this grid. These log
differences are not calibrated significances or refined shear estimates.

Passing seed/draw stability remains insufficient for absolute numerical
accuracy. At B's grid node(-0.00625952,-0.03264758), the exact repair changes
the curve by0.25055 log units, while the old variants differed from their
primary by at most0.02088 there. The sampled variants shared an error that
their mutual agreement did not reveal. Only the repaired four rows are
exhaustive; this cannot certify the remaining124 rows or all untested shear
points. No accepted shear estimate or0.3% calibration result follows.

## Expanded-step forensic audit on September30

The owner asked why adding bright atoms gives full matched-mock proposals
near 0.188 and 0.147. A new saved-array audit traces the arithmetic and
individual contributions without changing the inference or launching another
large likelihood calculation. Active source is
`scripts/audit_v39_expanded_step.py`; its frozen copy, CPU resource receipt,
`result.json`, and `exact_probe_replay.json` are under
`inference_v39_expanded_step_audit_20260930_v1` in the external catalogue-prior
root. CPU job 16773458 completed in 19 seconds. No GPU allocation was needed.

### Arithmetic, sample and normalization checks

The calculation uses all 16,314 retained matched-input mocks, including 134
with measured magnitude below 20. It excludes the 3,686 flagged numerical
padding rows from the original 20,000-row execution layout. The fixed centre
is (0.009836059884222188, -0.00029219982162802493), and the old sampler uses
K=1,024 Gaussian-ranked candidates plus a tilted sampled complement. The
large draws in this comparison are complement draws, not a larger exact
candidate set or a denser prior.

The audit verifies 50 source hashes, complete worker coverage, primary versus
maximum-rung equality, and all six aggregate scores and information matrices
for both mocks and actual measurements. Independent matrix solves and
influence-covariance calculations reproduce the recorded mock proposals and
conditional errors. Independent finite differences of the saved mass nodes
reproduce selection normalization derivatives. No factor-of-ten, padding-count,
or sign error was found in this saved-moment reduction. This does not validate
the numerical likelihood evaluations that produced the moments.

With S the summed selected score and I the summed observed information, the
reported proposal is g_center + I^(-1) S. The score changes much less than
the curvature:

| Complement draws | Summed g1 score | Minimum information eigenvalue | g1 proposal | Conditional g1 influence SE |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 12,179.13 | 747,240.09 | 0.01905090 | 0.00684 |
| 8,192 | 15,237.94 | 83,315.61 | 0.18813134 | 1.40956 |
| 16,384 | 16,189.03 | 112,803.98 | 0.14730972 | 0.90423 |

The final step has length 0.14356, 143.6 times the h=0.001 derivative stencil.
These are unsupported local extrapolations, not evaluated likelihood maxima.
The conditional errors are themselves unstable local diagnostics; they must
not be read as calibrated confidence intervals around an accepted shear.

Removing the normalization terms algebraically gives final g1=0.12554,
still a large proposal. This is an attribution check only: the scientific
likelihood must retain selection normalization. Its fixed derivatives cannot
cause the change across draw counts, though inversion amplifies their effect
when the information becomes small.

### Why many observations do not stabilize this solve

Projecting every final information matrix onto the weakest direction of the
summed matrix gives positive contributions totaling +5,200,966.42 and negative
contributions totaling -5,088,162.44. Their residual is only +112,803.98,
1.096% of the sum of absolute contributions. The matrix condition number is
about 15.3, so the issue is uncertainty and cancellation in the estimated
derivatives, not machine-precision instability in the 2x2 linear solve.

| Measured magnitude | Observations | Summed g1 score | Information along final weakest direction |
| --- | ---: | ---: | ---: |
| Below 20 | 134 | 2,598.42 | -227,704.62 |
| 20–22 | 774 | 13,109.32 | +16,130.12 |
| 22–24 | 4,164 | 297.61 | +307,172.57 |
| 24–25.8 | 11,242 | 183.67 | +17,205.91 |

Thus 80.98% of the final g1 score comes from magnitude 20–22, while bright
rows strongly alter the curvature dividing that score. Observation count
alone does not protect this estimator. These are contributions to the full
sample, not valid separately normalized magnitude-bin estimates.

For two rungs a and b, let x_a=I_a^(-1) S_a. Exact additive row contributions
to the step change are

`delta_x_i = I_b^(-1) [delta_score_i - delta_information_i x_a]`.

The row sums reproduce every audited step change. From 512 to 16,384 draws,
the total g1 movement is +0.12825883. The 134 bright rows contribute
+0.10591394, or 82.58% of that net movement. Splitting by changed score and
information gives +0.03852407 and +0.08973476, respectively. These decompositions
describe numerical change at fixed observations and prior; they are not a
physical population-effect estimate.

The smaller net 8k-to-16k movement, -0.04082162, also hides large cancellation.
For example, mock row 4460 contributes +0.33354 and row 2941 contributes
-0.28187 to that exact additive decomposition. These are contributions in
the common decomposition, not independent leave-one-out estimates. A small
aggregate difference could therefore hide much larger opposing errors.

### Extra atoms are present, but sampled curvature can misrepresent them

The three existing exhaustive worker outputs were independently recombined
and hash-checked, including complete 28,012,032-atom coverage and original
and expanded selection normalizers. This replays the earlier two stress
probes, not a new likelihood scan. For mock row 4460 (measured magnitude
18.4182), the g1 score and information at the shared centre are:

| Evaluation | Score | Information |
| --- | ---: | ---: |
| Original prior, exhaustive | +770.05 | +44,799.32 |
| Expanded prior, exhaustive | +97.91 | +165,073.89 |
| Expanded prior, sampled 8k | -91.00 | -30,543.77 |
| Expanded prior, sampled 16k | +591.53 | -233,919.29 |

Valid added bright atoms supply 86.246% of this object's exact centre
posterior mass; outside-domain added atoms supply 3.431%. The extension
therefore changes its actual likelihood substantially. The sampled curvature
has the wrong sign, with a final g1 information-entry error of -398,993.18.
This specific disagreement is numerical, since both calculations use the
same finite expanded prior. It does not establish whether the original or
expanded prior gives the better full-sample shear maximum.

The exact expanded posterior nevertheless has distinct-atom ESS 1.662 and
largest atom mass 76.70%. More total atoms did not make this local mixture
broadly supported. In the other exhaustive probe (actual row 1958), local
negative curvature survives exact summation. Finite-prior resolution and
nonlinear likelihood structure remain concerns alongside demonstrated
integration errors.

### Consequences for the next experiment

The 0.147 and 0.188 proposals cannot establish that extra bright support
failed, that the full likelihood favours extreme shear, or that all of the
16,314 observations carry consistent evidence for it. The strongest direct
finding is unreliable numerical curvature amplified by a Newton solve.
Prior-resolution sensitivity has separate supporting evidence from the A/B
profile comparison, with the limitations recorded above.

Keep the distinct matched-mock conditioning issue: these inputs were already
image selected before the new forward draw and selection. Correct integration
alone would not turn that experiment into unselected-parent closure. The
fresh-parent cohort is the cleaner control. A bounded comparison should
evaluate likelihood profiles accurately on fixed observations, include the
influential magnitude 20–22 range, and compare original and corrected expanded
priors before investing in another large mock run. No new profile, accepted
shear estimate, or 0.3% calibration claim is produced by this audit.

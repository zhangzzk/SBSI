# Paired positive- and negative-shear measurement inference

## Completed paired result — October 30, 2026

Final job 17613349 completed 0:0 at 19:02:46 Europe/Berlin. All 1,043 minus
C1 units and 311 C2 units completed, with no failed claims or omitted
measurements. All four C2 workers completed 0:0. The bounded monitor recorded
`result_written` and stopped; no user jobs remained at the completion check.

| Quantity | Estimate | Quoted uncertainty |
| --- | ---: | ---: |
| m1 | −5.539104% | 0.519981 percentage points |
| c1 | +0.000535043 | 0.000435911 |
| c2 | +0.001014857 | 0.000433398 |
| g1-to-g2 response | +0.008461197 | 0.004973010 |

These use the predeclared [paired definitions and uncertainty calculation](#estimands-and-uncertainty).
The paired case-bootstrap error on m1 is 0.463871 percentage points; the
galaxy-influence error is 0.519981 points and sets the quoted error. Both
include shared normalization uncertainty. The independent chain-plus-selection
floor is 0.179645 points. Pairing cancels much of the shared shape scatter;
its smaller slope uncertainty must not be replaced by an independent-leg
quadrature calculation.

Final plus shear is `(0.019427222142569794, 0.0011840807891115796)`;
final minus shear is `(-0.01835713610926805, 0.0008456329024757582)`.
There are 999,285 plus and 999,299 minus measurements, with 981,640 shared
identities and independent measured selection on each leg. Scalar recomputation
from these endpoints reproduces all four reported paired parameters within
1e-14. The paired JSON SHA-256 is
`8f87c7947596008ee72b87e1b04a5637d2cc737d93439df078932885764df3e4`.

The m1 deficit is 10.65 quoted standard deviations from zero; c1 is 1.23,
c2 2.34 and cross response 1.70. This is evidence of a substantial calibration
failure in the current model-plus-inference pipeline, not evidence that the
flow alone causes it. Finite-chain bias remains unresolved: minus C1 has
82,952/999,299 (8.30%) log-Z R-hat flags at >=1.1, and minus bright C2 has
5,789/18,929 (30.58%); plus has comparable flags. The quoted statistical errors
do not bound such bias. This reused development result does not establish
independent acceptance or 0.3% accuracy.

## Purpose and fixed endpoints

On October 29, 2026 the owner authorized inference on both measured shear
legs. The completed positive leg is frozen at
`galsbi_constgold1m_signsym_20261028/recovery_v1` under
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/`.
It contains **999,285 final measurements in 100 cases**, from one million
sampled raw rows. Its final estimate SHA-256 is
`1e743366386503a12b9830c2aafd9b25d25eb5241fcee8bb66021f6081a0af5b`.
The new negative leg is `galsbi_constgold1m_minus_signsym_20261029` under
the same parent. No positive-leg chains are rerun.

Preparation 17600115 completed successfully in 1m39s: **999,299 minus
measurements** from 1,000,052 sampled raw rows, with 753 removed by final
support cuts. The selected populations share 981,640 identities; 17,645
are plus-only and 17,659 minus-only. Of the latter, 17,393 were absent
from the plus base-eligible population, and 266 belonged to the raw plus
sample but failed its final support cuts. Minus C1 contains 112,318 bright
and 886,981 faint galaxies in 1,043 units; C2 reruns 18,929 mag<21 galaxies
in 311 units.

This varies injected shear between `(0.02,0)` and `(-0.02,0)`, unlike the
earlier model-generated rotated-pair test, which kept injected shear fixed.
See [the endpoint audit](MOCK_MEASUREMENT_REVIEW.md#conclusion-and-scope).

## Population and paired sampling

Read actual `case40` through `case139` negative-shear measurement catalogues.
Match underlying identities by `(source_case, source_input_index)`, and
verify equality of unsheared position, redshift, size, axis ratio,
position angle, Sersic index and magnitude across the input catalogues.
Detection IDs are local to each leg. Reject duplicate identities; report
unmatched catalogue rows. Apply each leg's own detection/usability and
measured cuts: `MAG_AUTO<25.8`, `FLUX_RADIUS>3` pixels (0.6 arcsec), then
the inference support bounds `FLUX_RADIUS<100` and `|e|<0.99`. No
truth-property population cuts. Shape and selection definitions follow
[Conventions sections 3–5](CONVENTIONS.md#3-population-and-selection-order).

The plus raw draw is a uniform one-million-row sample from 18,157,963
base-eligible rows, before its last support cuts. Let
`q=1,000,000/18,157,963`. A minus-base-eligible identity that was also
plus-base-eligible is included exactly when it belonged to that frozen raw
plus draw. A minus-base-eligible identity absent from the plus-base-eligible
population is included independently with probability `q`.

Thus every minus-eligible identity has the same marginal inclusion
probability. This preserves pairing where possible **without conditioning
the minus population on plus detection or final selection**. In particular,
the 715 raw plus draws removed by its final support cuts remain eligible
for the minus draw. Final minus count is determined by its own measurements,
not forced to one million. Both populations and their overlap are reported.

The new Bernoulli seed is 2026102901, split by case; ordering seed is
2026102902. Preparation exactly replays the plus source hashes, eligibility
reports, sampled IDs and measurement values before freezing minus inputs.

## Model, inference and normalization

Use symmetric F8 `f8_signsym_20261026_v2/base/flow/guard/selected.pt`, old C8,
and the unchanged physical disk emulator. Artifact identities are frozen in
the run's `rule.json`. The validated recovery sampler evaluates the
likelihood for every proposal and audits retained states; it does not reuse
likelihoods across redshift/SED proposals.

Every minus chain is new. Preserve 20,480 steps, 8,192 warmup, stride 16,
four chains, original likelihood grouping, starts, temperatures and prior.
C1 samples at zero shear; C2 reruns `mag<21` at the rounded C1 shear.
C1 seeds/bank seeds start at 2028700001/2033700001; C2 at
2028800001/2033800001, each plus the declared unit index. These differ from
the plus seeds. The plus endpoint retains its historical mixture of cached
C1 and repaired full-likelihood C1 units; its final C2 is full-likelihood.

Reuse the same candidate-specific 10,010,624-scene normalization from
`galsbi_constgold100k_signsym_20261027/preliminary_10m`. The model, prior,
observing conditions and measured selection are unchanged. Injected truth
does not enter the likelihood or change its selection integration grid.
Propagate this shared normalization uncertainty jointly between legs.

## Estimands and uncertainty

For estimates `g+` and `g-`, report

- `m1=(g1+ - g1-)/0.04 - 1`;
- `c1=(g1+ + g1-)/2`;
- `c2=(g2+ + g2-)/2`;
- cross response `R21=(g2+ - g2-)/0.04`.

These are a central slope and even intercept at these two shears, with the
usual linear-response interpretation. They do not measure `m2`, and cannot
separate curvature from the even intercept. Report `m1` in percent with
errors in percentage points; stored values are dimensionless. This is an
inferred-shear calibration slope, not a direct forward-response ratio;
see [bias conventions](CONVENTIONS.md#7-multiplicative-bias).

Predeclare 2,000 shared-case bootstrap draws (seed 2026102903) over the same
100 cases, recomputing both ratios of summed score to information. Also
compute a joint galaxy influence covariance, joining the selected identities
across legs and giving absent identities zero influence. For each leg,
the influence is `I_total^-1 (v_i - I_i g_hat)`.

The shared selection-gradient covariance comes from all 1,222 normalization
chunks, retaining its off-diagonal element. Its leg response is
`-N I_total^-1`, holding information fixed as in the existing marginal-error
approximation. A separately reported chain-half covariance floor assumes
independent chain noise between legs. Quote the largest marginal uncertainty
among case-plus-selection, galaxy-influence-plus-selection, and
chain-floor-plus-selection. Do not add chain scatter again to empirical
bootstrap/influence scatter. Preserve all three covariance estimates.

This is known-shear development, not independent acceptance. No
finite-population correction is applied. Finite-chain bias is not bounded
by these errors; pairing does not cure the existing convergence flags.

## Execution and verification

Entry points are `scripts/run_sign_symmetric_minus.py` and
`scripts/paired_sign_symmetric_estimate.py`. Setup freezes scripts and
configuration; preparation freezes measured inputs and unit lists.
Preparation and reductions run through Slurm. GPU sampling depends on
preparation, the test suite, and short bright/faint/nonzero-shear integration
checks. Four standard `inter` workers and two 8-CPU A40-16GB `cip` workers
share atomic C1 claims, at most six GPUs. C2 starts only after all C1 workers
and its reduction finish, then uses four `inter` workers. Each unit has
one identical retry with its failed attempt preserved; failed selected
units are not silently omitted.

The first preparation (17599986) passed all 100 catalogue/identity checks,
then failed before writing observations because the new adapter omitted the
derived `measured_shape_log_disk_gap` column. Added the identical
`log1p(-e1**2-e2**2)` encoding used by the frozen plus adapter; Cartesian
measurements are unchanged. Added a regression test; all ten targeted
adapter/paired tests passed. The failed attempt, initial job list and source
manifest are retained, and its unused dependent checks were cancelled.

Replacement jobs: preparation 17600115, preflight 17600119,
C1 workers 17600120/122/124/126/128/130, C1 reduction 17600132.
Full test job 17599987 passed (729 passed, 4 skipped) before the repair;
the repair then passed the targeted suite. Full IDs and future C2/final
jobs are in `jobs.json`. Two-hour bounded
checks write on-disk status receipts and stop on failure or the final
paired result; they do not send chat messages. Replacement monitor 17600133.

Preflight 17600119 completed 0:0 in 2m13s. Bright, faint and nonzero-shear
checks evaluated 128, 16 and 128 retained states respectively, with zero
summary or log-kernel replay differences and no skipped likelihood calls.
These short chains test integration, not convergence. Both cip workers
17600128/130 then passed their own faint integration checks and started
six MPS processes each on cip-cl-h01g02n2/n3. The four inter workers remain
queued as of startup.
All first 12 production units subsequently wrote sampling progress (about
128/20,480 steps), with no failed claims. The frozen run's
`startup_receipt.json` records integration audits, first progress and scheduler
state. This completes startup monitoring; bounded later checks remain queued.

At submission nine GPU slots were unallocated: four A40-8GB, two
A40-16GB, one H200 with only two free CPU cores, and two RTX2080Ti with
ten free CPU cores total. Only the two A40-16GB slots immediately fit our
validated small-worker configuration. The four standard workers must queue.

# V3.9 mocks from the exact actual20k galaxy inputs

## Owner request and generation

On2026-09-27 the owner requested model-generated mocks using exactly the
inputs associated with the failed actual20k run, to test whether they show
the same first-pass curvature problem. The owner explicitly chose a normal
forward draw, accepting fewer retained observations after model selection.

The20000 ordered(case,input_index,detection_id) identities come from the
[actual ConstGold plus-leg run](V39_ACTUAL20K_BFGS.md#request-and-sample).
Each is joined to its original input catalogue, with no new truth-property
cuts, replacement identities or top-up. All rendered neighbours remain
available when constructing features, including galaxies outside the20k
target list. Missing or duplicate target identities cause rejection.

For each input, the generation is:

1. Derive intrinsic ellipticity from its original axis ratio and position
   angle, retaining pre-shear circularized radius. Construct the shared F8/C8
   physical crowding summaries over all nonself neighbours within10arcsec.
2. Apply shear(0.02,0) to target ellipticity, then draw one Bernoulli usable
   event from the pinned C8 classifier.
3. For each usable input, draw one joint F8 measurement of shape, radius and
   flux. The same radius/flux conditions the original physical-moment disk
   emulator. Sum its up-to20 neighbour coefficients and apply Möbius shape
   transport. The emulator and F8 crowding use their distinct pinned geometries.
4. Apply the newly generated measurement cuts: radius>3pixels(0.6arcsec),
   magnitude<25.8. Retain every passing draw; do not redraw rejected inputs.

Usability and flow seeds remain202609282 and202609283, respectively. No
prior-atom sampling occurs in generation. The inference still integrates
over the unchanged20,231,221-atom restored prior; generating truth inputs
are not added to it. Every attempted input's usability/selection outcome,
all usable draws, and the retained catalogue are saved with stable IDs.

## Controlled curvature comparison

All likelihood evaluations use the actual run's failing expansion point
(0.009836059884222188,-0.00029219982162802493), with its verified full-prior
normalization. No new shear centre or model is fitted. Holding the point
fixed avoids attributing a change in curvature to a different evaluation
location. The mock's own raw mean is recorded for context only.

The comparison contains three sets: generated measurements that pass the
new selection; actual measurements for exactly those retained IDs; and the
original full20k actual result. The second set separates changed membership
from changed measurements. Its reduction reuses the saved actual moments.

Inference keeps the Gaussian131072-prefilter/top1024 proposal, seed8701,
tilted-stratified delta0.1, h=.001, full two-component information and nested
draw ladder512–16384. Proposal draws use original20k parent row IDs, not
new positions after mock selection.

Original groups of four observations are preserved. For numerical evaluation
only, nonretained mock IDs are padded with their saved actual measurements.
They are explicitly flagged and excluded from every mock score/information
sum. Thus the worker evaluation table has20000 rows while the scientific
mock catalogue contains only the new selected subset. This preserves the
known grouping constraint without treating padding as synthetic observations.
An exact replay compares all ten arrays from original versus batched
retrieval/draw preparation before production. Likelihood density batches
remain four observations.

## Execution and validation

Frozen root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_constgold20k_matched_mock_20260927_v2`.

CPU preparation reconstructs all100 source cases, checks original input
hashes, and replays overlapping features against the pinned V3.9 development
contexts for cases40–119. The shared feature helper has tests for ordered
identity matching, missing/duplicate rejection, no truth cuts, complete
neighbour support and exclusion of numerical padding from retained IDs.

Jobs: preparation16729151; full tests16729152; single-GPU generation16729153;
GPU replay16729154; four persistent inference workers16729155[0–3]; CPU
comparison16729156. Dependencies and array throttling keep the run within
four concurrent GPUs. GPU workers have a two-hour time limit; the expected
first-pass evaluation is approximately30–40minutes after worker startup,
with no new full-prior normalization stage.

The first setup(v1; preparation16729136 and downstream jobs16729138–41)
was cancelled before any mock generation to preserve original20k density
groups after mock selection. Its partially prepared files and logs remain;
v2 is the active run. No numerical results or seeds were selected between
versions. The first setup's focused tests passed15 cases, and its full suite
passed316 tests with3 optional skips; v2 reruns tests including the added
padding invariant and with model artifact roots supplied.

v2 validation:16 focused tests passed; the full suite passed319 tests with
one optional skip in26.80s. Preparation completed all100 cases in108.8s,
with20000 matched IDs and no missing rows. The7980 overlapping target-role
contexts in cases40–119 replayed within the declared float32 tolerance.
Intrinsic ellipticity reconstructed from axis ratio/angle matches the raw
recorded intrinsic components to7.50e-16 maximum absolute difference.

Generation completed in10.2s:18598 usable draws,2284 failing new measured
cuts, and16314 retained mocks.1402 inputs failed the C8 Bernoulli draw.
There are134 bright mocks and134 bright actual observations among the same
retained IDs. The mock raw mean is(0.0125365314,-0.0003598444); the actual
mean for those IDs is(0.0108274212,-0.0015714794). Neither replaces the
predeclared shared expansion point. Numerical padding comprises3686 rows
and is excluded from the mock reduction. GPU replay16729154 passed exact
equality on all ten arrays for128 original-group rows in233.6s of Python
execution. All four workers16729155[0–3] subsequently completed successfully
in30m30s–30m45s, followed by CPU comparison16729156 in20s. Each worker
evaluated5000 original rows; the reduction excludes all3686 padding rows.
There are no failure receipts. The redundant pending status-only audit
16729189 was cancelled after the owner's completion check; status_latest.json
was refreshed directly. These snapshots do not wake an agent or notify the owner.

## Completed comparison on September27

The matched-input mocks reproduce the actual run's indefinite observed
information at the shared centre(0.0098360598842,-0.000292199822).
At16384 draws per observation, the summed information eigenvalues are:

| Measurement sample | Selected observations | Minimum eigenvalue | Maximum eigenvalue |
| --- | ---: | ---: | ---: |
| V3.9 mocks from exact actual inputs |16314 |−1767373.8285 |2002216.2944 |
| Actual measurements for the same retained IDs |16314 |−2394269.7775 |3423542.5435 |
| Original actual measurements |20000 |−2387169.1061 |3445424.7863 |

All three matrices are indefinite at every tested draw count:512,1024,
2048,4096,8192 and16384. This is a successfully completed diagnostic whose
curvature fails the initial Newton acceptance check, not an execution failure.
No accepted Newton shear estimate, bright-only continuation or bias estimate
was produced. Full scores, matrices and draw-count comparisons are retained
in comparison.json under the frozen root above.

Thus actual-image/model measurement mismatch is not necessary to reproduce
this failure: it also occurs using the model's own generated measurements
on these inputs. This does not establish that the measurement model is
accurate, nor identify a unique cause. Earlier prior-generated mocks drew
their generating atoms from the same finite prior used by inference;
these actual galaxy inputs have no such support guarantee. Those earlier
bright-only iterations settled but did not pass the draw-convergence check.
Finite-prior support, correlated integration error and local non-concavity
remain unresolved possibilities. A targeted bright likelihood profile with
independent integration checks is the next useful diagnostic; none was
launched as part of this completion check.

## Expanded-prior comparison requested September27

After the completed failure comparison, the owner requested the prior with
the extra bright atoms. The new run reuses the exact saved16314 mocks,
their original20k evaluation groups and all20000 actual measurements. No
new mock draw, identity selection, truth cut or shear-centre change occurs.
Both measurement sets are reevaluated under the expanded prior; the actual
likelihood moments cannot be reused after changing that prior.

Restore all7,780,811 added atoms, giving28,012,032 total, with the existing
[bright-extension weights and environments](BRIGHT_PRIOR_EXTENSION.md#weights-neighbours-and-identities).
Nonbright weights and total bright probability remain unchanged. Use the
V3.9 feature/support/coordinate caches from
`inference_v39_mock500k_distance_20260926_v2`, with the same frozen runtime
as the just-completed matched-input run. This does not use the old V3.6
17-feature store as a V3.9 feature cache. Preparation verifies all original
atom features and proposal coordinates/floors exactly against the current
20,231,221-atom run, plus nonbright weights, total mass and bright weights.
The proposal metric remains fixed. Old proposal C8 base probabilities are
retained; added atoms get C8 evaluated at the same cached reference point,
zero shear. This proposal probability point differs from the fixed
likelihood expansion point.

Selection normalization is recomputed over all28 shards at the shared
actual-data centre and its h=.001 stencil, using64 CRN QMC draws per atom
and the existing per-shard seeds. Each of four persistent GPU workers
processes seven shards, then waits at a checked barrier. Worker0 assembles
normalization and verifies exact original-versus-batched proposal replay
for both observation sets, including bright groups. After this gate, each
worker evaluates its original5000-row block for mocks and actual data.
Density groups stay at four; proposal preparation stays at24. Final reduction
compares both priors at every draw count512–16384 and reports the paired
8192-to16384 change where positive information allows an estimate.
An indefinite matrix is recorded as a diagnostic outcome, not repaired.

Frozen root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_constgold20k_matched_expanded_20260927_v1`.
Driver: `scripts/run_v39_expanded_prior.py`, importing the explicitly pinned
matched-input and recenter drivers. Tests16729474 and CPU initialization
16729457 gate four GPU workers16729459[0–3] on inter/th-cl-nv01. No other
SBSI GPUs were allocated at submission. Initialization passed in25.8s of
Python execution. Expected wall time is approximately1.5–2hours after GPU
startup for normalization and both measurement sets; jobs have a4hour limit.
No BFGS continuation or production-prior default change is part of this test.
Initial tests16729456 on cip-cl-compute4 stalled in Python import filesystem
I/O (`cl_sync_io_wait`) before emitting any pytest output. Cancelled that
attempt after moving the worker dependency to the identical replacement
suite16729474 on inter/th-cl-1024us01. Scheduled status-only audit16729469
about two hours later; it saves status_latest.json without waking an agent.
Replacement tests passed319 cases with1 optional skip in27.28s. All four
workers started, and each completed its zero-shear and central normalization
nodes for its first shard with finite positive masses (about0.01288–0.01293).
No failure receipts. The run now continues through full normalization,
the automatic replay gate and both likelihood evaluations.

At the owner's next status check (56minutes elapsed), all28 normalization
shards and assembly are complete. Exact replay passed all ten arrays for60
mock and64 actual rows, including bright groups. Twelve of20 mock evaluation
chunks are saved and all workers have entered their fourth chunk; actual
evaluation follows. No failures or final comparison yet. Recent1000-row
chunks take346–361seconds, giving an estimated45–55minutes remaining.

## Completed expanded-prior result on September27

All four16729459 workers completed with exit0 in1h39m41s–1h41m24s;
worker0 saved the final comparison. Status audit16729469 also completed.
All eight mock/actual worker receipts and moment-file hashes were verified
at the owner's completion check; no failure receipts or active jobs remain.
Original20k group coverage and removal of3686 numerical padding rows are
checked by the reducer. Scientific sample sizes remain16314 mocks,
16314 matched actual measurements and20000 original actual measurements.

At16384 draws, the smallest summed observed-information eigenvalues are:

| Measurement sample | Original prior | Expanded prior |
| --- | ---: | ---: |
| Matched-input mocks |−1767373.8285 |112803.9799 |
| Actual, same retained IDs |−2394269.7775 |−995216.0549 |
| Actual, all20k |−2387169.1061 |−995018.5262 |

The mock matrix becomes positive definite, but its Newton proposal remains
unstable. The fixed-centre mock g1 proposals across draws are:

| Draws per observation | Mock g1 proposal |
| ---: | ---: |
|512 |0.01905090 |
|1024 |0.01931550 |
|2048 |0.02426475 |
|4096 |0.05516627 |
|8192 |0.18813134 |
|16384 |0.14730972 |

At16384, the full proposed shear is(0.14730972,0.04107400), with a
step norm of approximately0.14356 from the unchanged expansion point.
This would exceed the historical0.05 recentering guard. The reported
conditional sandwich standard errors are(0.90423165,0.27985933);
these large local diagnostics do not establish convergence or calibration.
No continuation was attempted. The metadata flag
same_problem_at_shared_center=false only means that the final mock matrix
is positive; it does not mean the inference problem was solved.

Both actual samples have indefinite information at512,1024 and2048 draws,
positive information at4096 and8192, then indefinite information again at
16384. For all20k actual rows, the minimum eigenvalue changes from346256.2484
at8192 to−995018.5262 at16384. Choosing a lower draw count because its
proposal looks closer to injected shear would be outcome-based selection.

The expanded prior substantially changes these finite-draw scores and
curvatures, but has not recovered reliable inference or established0.3%
accuracy. A prior change also changes retrieval/proposal draws, so these
results alone do not isolate improved physical support from integration
error or true local non-concavity. Bright likelihood profiles and stronger
independent integration checks remain the appropriate next diagnostic;
no additional run was launched at this completion check.

The September 30 [expanded-step forensic audit](V39_POPULATION_AUDIT.md#expanded-step-forensic-audit-on-september30)
reproduces these proposals from the saved moments and quantifies the strong
cancellation in their curvature. It also replays the exhaustive probe where
the sampled expanded-prior curvature has the wrong sign. The large proposals
are not evidence that the full likelihood prefers shear near 0.15 or 0.19.

### Subsequent population and integration audit

The owner's follow-up prompted a separate
[population audit](V39_POPULATION_AUDIT.md#original-prior-versus-the-actual-unselected-parent),
including all100unselected input cases and weighted priors. It identifies
a specific generator-domain mismatch in the added bright support, while
the original prior closely matches the unselected intrinsic-property and
crowding histograms. See the
[conditional-mock distinction](V39_POPULATION_AUDIT.md#why-the-matched-input-mocks-have-a-different-target-distribution)
and [exhaustive likelihood probes](V39_POPULATION_AUDIT.md#exhaustive-likelihood-probes)
before interpreting reweighting as a remedy. No existing comparison result
or production weight is replaced by that audit.

## Simulation role coverage

The exact actual20k cohort includes9979 target-role inputs and10021 inputs
outside the structural target role used by the V3.9 development evaluation.
The role definition is original input_index<floor(N_generated/2), per case;
it is not inferred from measured properties. This also explains why the
cached target-role feature replay covers only7980 of the inputs in the80
overlapping development cases. All20000 raw identities are matched and used.

| Original simulation role | Input attempts | Model usable | Model selected |
| --- | ---: | ---: | ---: |
| Target role |9979 |9257 |8118 |
| Outside target role |10021 |9341 |8196 |

`input_role_audit.json` records this split against the original catalogue
row counts and generated outcome hash. It adds no cut and changes neither
the mock nor actual sample. Role coverage is an interpretation issue to
investigate, not an established cause of the curvature failure.

## Interpretation limits

These inputs were selected through the original image measurements, then
subjected to a second model-generated selection. Their distribution is not
the original unselected simulation population. This is a matched-input
diagnostic, not an unbiased population-closure test, independent validation
or robust0.3% claim. A one-step estimate, if its curvature is positive, is
only a local diagnostic; no BFGS continuation is part of this comparison.

The observed indefinite mock curvature shows that the problem can persist
with model-generated measurements from these inputs. The comparison is at
the original actual sample's fixed expansion point; it does not establish
curvature at the mock's own mean or global optimum. It cannot by itself
separate finite-prior support, stochastic integration, local curvature and
model error.

# Flow-training self-response, detection and selection

## Covariance grid216 flow replacement — 2026-09-12

The owner requested a remake of the B/U/p training-scene table for the new
flow. This run uses the completed circularized 6x6x6 grid shape+radius
lambda1 **covariance-weighted epoch146** checkpoint, SHA256
`0e9bf0afbdd4833bbffb9aa486903ee99b0c8cc079990c5a1712b55a6323b37a`.
It does not use the still-running no-covariance companion. See
[the flow trial](FLOW_CIRCULARIZED_GRID_SHAPE_RADIUS.md#covariance-branch-final-review--2026-09-12).

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/training_self_selection_grid216_cov_lambda1_v1`.
The frozen `manifest.json`, `workers.json` and `runner/worker_hashes.json`
record the replacement and original evidence; `submission.json` tracks jobs.
Original prepared case directories are reused through a symlink and verified
against their existing hashes. No catalogues, measured moments, exclusions,
detector, neighbour features, cuts, seeds, response ruler or case membership
are changed. No R_blend is applied. The evaluation explicitly overrides the
archived evaluator's selected flow with this new checkpoint; case receipts
record its hash, the new manifest hash and the prepared receipt hash.

The same160 training scenes contain7,770,008 physical truth objects,
6,943,102 B usable pairs and6,834,259 BT cached NLL pairs. All13,821,950
original NLL rows were reproduced during the original preparation. Retain
the six previously reported unmatched crowd IDs (three with nonzero shear)
and all complete neighbour scenes. The detector probabilities must reproduce
exactly for every case. This remains an in-sample forward0-to0.05 primary
parallel/cross diagnostic, with m=R_measured_parallel/R_model_parallel-1,
pooling moments before ratios, case bootstrap10000 seed20260902 and separate
integration uncertainty from16 independent64-draw groups seeds7301–7316.

The inherited workers evaluate all8 cut combinations and15 preset thresholds,
including the requested none / magnitude+radius / all columns for B/U/p.
BT/T and exact true-plus-residual identities remain diagnostic controls.
`runner/compare.py` additionally checks identical measured sufficient sums
and detector probabilities, computes paired new-minus-old m uncertainties
using the same cases and latent groups, and writes `comparison.json` and
`table.md`. Final `completed.json` is written only after this comparison;
the inherited review's intermediate receipt is `review_completed.json`.

CPU gate16467405 checks all prepared hashes/counts, identical condition
preprocessors and loading of the new checkpoint. GPU control16467406 first
replays old-flow case0 seed7301 against the archived selected moments for
all five parents and23 cuts, then evaluates all16 groups with the new flow.
Array16467407 evaluates indices1–159 (manifest cases[index]), concurrency4,
after the complete control; dependent review16467408 includes case0 once.
Heavy work uses Slurm. AST and Bash syntax checks passed. Startup and final
review findings will be recorded here; submitted job states are not results.

Startup gate16467405 completed in82s with exact preprocessor equality,
successful model loading and all160 prepared hashes/counts verified.
Control16467406 completed in235s: archived case0 seed7301 reproduces with
maximum sufficient-statistic difference exactly0 across all five parents
and23 cuts; detector probabilities also reproduce exactly. The new-flow
case0 finished16groups/99,246,080 finite draws, retaining90 numerical
zero-radius values and0zero-flux values without corrections. All output
hashes and selected-checkpoint hash passed. These rare zeros are a support
limitation, not discarded draws. Array cases1,2,3,5 have each completed
their first sampling groups; every startup stderr is empty.
`startup_review.json` records this evidence, without an interim m estimate.
A self-removing one-shot is due2026-09-12T21:19:51UTC, tag
`sbsi-grid216-self-selection-once`; unrelated cron and the separate
no-covariance monitor are preserved. App automation tool discovery returned
no callable tool, so the existing Slurm/Codex queue wrapper was reused;
syntax and premature-inert checks passed. Final160-case review is pending.

The scheduled bounded check at2026-09-12T21:20UTC found128/160 completed
case receipts, four GPU tasks progressing and review16467408 still pending.
Every existing stderr was empty. Sampled completed cases0/163/164 have all16
seeds, finite receipts and matching checkpoint/manifest/prepared-receipt
hashes; all14 frozen worker/manifest/launcher hashes also match. This is a
progress check, not the final aggregate comparison; no interim m is reported.
The first one-shot remains disabled. Its continuation,
`one_shot_check2/watch.json`, is due2026-09-12T23:21:46UTC and uses the same
`sbsi-grid216-self-selection-once` tag. Wrapper syntax and premature-inert
checks passed; unrelated cron, including the no-covariance check, is preserved.
See `one_shot_check2/bounded_review.json` for the compact receipt evidence.

### Completed covariance-grid replacement result — 2026-09-12

All160 case receipts and dependent review16467408 completed successfully;
review runtime was5m55s with empty stderr. Every case contains all16 frozen
QMC groups. The final review reproduces original measured sufficient sums and
all detector probabilities exactly, verifies every checkpoint/manifest/input/
output hash and recomputes `m` from pooled responses. A separate post-run check
recalculated all completed output hashes and each displayed response ratio.
The now-unneeded scheduled check was disabled and only its tagged cron line
removed; the usable-event trial monitor remains active.

| Population | No measured cuts | Magnitude + radius | All cuts |
|---|---:|---:|---:|
| Fixed both-usable `B` | +0.643 +/-0.149% | +0.345 +/-0.138% | -1.517 +/-0.218% |
| Actual per-leg usable `U` | +0.164 +/-0.150% | **-0.025 +/-0.139%** | -1.959 +/-0.219% |
| Raw-`D` classifier `p` | +1.795 +/-0.154% | +1.909 +/-0.140% | -0.431 +/-0.223% |

Errors are case-bootstrap SE; integration SE is only0.0009--0.0036percentage
points. Compared with the old lambda10 flow under exact shared cases, detector
and CRN, the new-minus-old changes for `(none,no-e,all)` are
`(-0.189,+1.563,+0.592)`pp for B, `(-0.263,+1.479,+0.521)`pp for U and
`(+0.291,+2.081,+0.668)`pp for p, with paired case SE at most0.0145pp.
The flow change is real, not MC noise.

The tempting all-cut p result is not evidence of0.3% closure. It is still
outside the central-value specification, and its95% interval is
[-0.871,+0.005]%. More importantly, it is a cancellation: p is+1.909% before
the measured ellipticity cut, and that cut shifts it by-2.340pp. The actual-U
all-cut comparator remains-1.959%. Likewise, substituting the learned raw-`D`
classifier for the actual per-leg mask changes the result by about
+1.630/+1.935/+1.528pp across the three columns. This contrast combines
classifier approximation error with the tiny physical `D`-versus-usable
difference; it must not be attributed to event semantics alone.

There is one genuinely promising but narrow result: actual U with magnitude
and radius cuts and no measured-|e| cut is-0.025% with case-bootstrap95%
interval[-0.297,+0.249]%, versus-1.504% for the old flow on these same scenes.
That satisfies the numerical band in this reused in-sample self-response
diagnostic. It does not establish generalization, coherent R_blend closure or
inference calibration: these are flow-training scenes, some overlap detector
training, no R_blend is used, and the covariance model has not been evaluated
on the coherent ConstGold endpoint. The measured-|e| cut alone moves U from
-0.025% to-1.959%, so retaining that cut is not a credible0.3% path with this
model.

Decision: do not promote the covariance-grid flow and do not cite its all-cut
classifier cancellation as success. Retain it only as a candidate component
for the explicitly changed no-|e| science sample. The direct usable-event
trial was subsequently invalidated because `_secondaries` bookkeeping was
misread as measurement failure. The next informative test is therefore a
held-out coherent no-|e| composition using this flow with the unchanged valid
raw-`D` detector and an actual-mask diagnostic. Failure there would reject the
covariance-flow candidate; it would not justify threshold tuning or an
empirical response correction.

### Coherent covariance-flow rejection — 2026-09-13

The prescribed held-out-in-flow-context coherent screen is complete on all100
cases40--139,9,709,701 objects and16x64 CRN groups. It changes only the flow
checkpoint; actual masks, current raw-`D` probabilities, trial9 `R_blend`,
cuts, support and estimator are identical. The old sufficient arrays reproduce
to maximum absolute error3.39e-11. Independent post-review reconstructs all100
receipts, every displayed point/SE/interval, and the full=true+residual identity.

| Parent | No measured cuts | Magnitude + radius, no measured \|e\| | All cuts |
|---|---:|---:|---:|
| Fixed both-usable `B` | +0.291 +/-0.152% | +1.831 +/-0.123% | +2.227 +/-0.217% |
| Actual per-leg usable `U` | -0.486 +/-0.158% | **+1.153 +/-0.127%** | +1.613 +/-0.222% |
| Current raw-`D` probability `p` | -0.699 +/-0.156% | **+1.017 +/-0.128%** | +1.854 +/-0.224% |

On no-|e| the paired new-minus-lambda10 change is+1.282pp for `U` and
+1.699pp for `p`, with paired case SE0.00568/0.00449pp; integration jackknife
SE is only0.00150/0.00149pp. The covariance p-model R11 is0.99731063, outside
the allowed `[1.00444145,1.01048623]` interval by0.00713081. Its truth and
residual model-minus-data errors are both negative(-0.002729 and-0.007415),
so the failure is not stochastic integration or opposing-component
cancellation. Although p and U are closer to each other under this flow, the
actual-U oracle itself moves from-0.129% to+1.153%; composition improvement
cannot rescue the wrong conditional moment.

Original array16468278 completed cases40--89 but exposed a storage-contract
error for cases90--139: their reviewed preparation stores `p` separately. Each
failed task drew one unsaved group and wrote no output. The repair reads the
already reviewed detector-probability artifact, verifies IDs plus artifact and
completed-receipt hashes, and records that provenance. Retry16468415 and
review16468416 completed cleanly. The retained original failure logs are not
scientific case errors. Final result/completed/post-review SHA256 are
`fe539340...49afe`/`54606a59...21f36`/`fa2f1f31...541b4`.

Conclusion: reject covariance grid216 lambda1 for this inference path and keep
lambda10 epoch154. The apparently excellent training-scene `U/no-e=-0.025%`
was indirect, in-sample evidence and failed the necessary coherent
generalization check. Further tuning this flow on cases40--139 would compound
endpoint reuse.

## Fixed diagnostic — 2026-09-12

The owner requested the same matched/actual-detection/classifier response
comparisons on flow training data, with self-response only. This tests whether
the ConstGold discrepancies also occur in the flow's training scenes without
a coherent-neighbour correction. No training, fitted correction, threshold
optimization, or model promotion is part of this experiment.

External experiment:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/training_self_selection_v1`.
`manifest.json`, `workers.json`, `setup_hashes.json` and `probe_hashes.json`
freeze inputs and workers; `submission.json` records jobs and replacements.
The frozen SBSI environment/source come from the lambda10 flow's
`constgold50_v1` evaluation. Production library files are unchanged.

Models are the lambda10 shape+radius circularized flow epoch154
(SHA256 prefix `9c5bb028437a`) and circularized-BCE detector
(`b1dfa6361a78`). There is **no R_blend prediction or shift**.

Use all160 cases from the original seed501 training split, checking all
13,821,950 NLL rows against the exact cached measurements and circularized
contexts. The expected nonzero-primary-shear cached intersection contains
6,834,259 pairs. Science responses exclude primary galaxies with zero native
shear, reporting those exclusions separately after checking their NLL rows.

The physical truth population is fixed across legs: strict18<r<25.8 and
0.37<semi-major Re<1.5arcsec, retaining the complete neighbour scene.
There is no new subsample or neighbour-distance cut on this population.
The detector was trained on cases0–24 and tuned on25–29, with test30–39;
This flow-training split overlaps19 detector training cases,4 tuning cases
and8 detector test cases. This is an in-sample
diagnostic, not an independent validation claim.

## Response and population definitions

Follow [CONVENTIONS.md §6b–6e](CONVENTIONS.md#6b-isolating-the-self-response-half-shear-only).
Native simulations use forward0→0.05 shear in an independent random direction
for each primary. On both sides, project ellipticity onto that primary's
unit shear direction and its perpendicular, then difference the two pooled
selected means and divide by0.05. Cuts use sky-frame measurements before
projection. Store sheared first, unsheared second. For a fixed matched
population this equals the mean paired projected difference. For populations
that change between legs, retain their separate normalization denominators.

Report `m = R_measured_parallel / R_model_parallel - 1`, with the parallel
and cross responses themselves. These are self-response closure proxies,
not coherent ConstGold R11 or actual SBSI inference m.

| Parent | Model evaluation population | Measured comparison |
|---|---|---|
| B | Fixed intersection of usable raw detections | Same fixed intersection |
| U | Actual usable detection flags separately per leg | Same separate flags |
| p | Classifier probabilities over full physical truth | Actual U |
| BT | Fixed intersection of exact cached NLL membership | Same cached intersection |
| T | Exact cached NLL membership separately per leg | Same separate membership |

Usable raw detections require a unique crossmatch, finite shape/magnitude/
log-radius, and valid ngmix flags. The raw crossmatched detection label D
is recorded separately from usable U, matching the detector's training label.
Unmatched, invalid, out-of-support, and zero-primary-shear counts are reported.

For each parent evaluate all8 combinations of |e|<0.6, MAG_AUTO<25.8 and
radius>=3.75pixels (0.75arcsec), plus the same15 preset thresholds as the
[ConstGold replication](CONSTGOLD_SELECTION_REPLICATION.md#fixed-comparison-and-estimators--2026-09-12).
The model draws and selects its own measurements. Integrate with16 independent
64-draw QMC groups, seeds7301–7316, using CRN across both legs and all cuts.
No additional explicit mock catalogue run is included; the response uses
integrated selected moments, as in the preceding B/U/p comparison.

Pool sums before ratios. Use paired case bootstrap10000, seed20260902,
and separately report integration uncertainty from deleting one latent group.
Retain the exact selected response identity
`R_measured = R_true + R_(measured-true)`, cut interactions, drop-one-cut
effects and paired B/U/p/T/BT contrasts. Conditional-covariance controls are
finite-MC products and Shapley allocations are accounting conventions.

## Catalogue reproduction and eligibility finding — 2026-09-12

Setup16463943 succeeded, sharding both original catalogue sources and all
13,821,950 NLL references. Probe16463971 succeeded on case0 and established
the raw shape filename required for exact reproduction:
`shape_catalogue_detect_position_secondaries_tile180.0_-0.5.feather`.
The sibling without `_secondaries` has different ngmix shapes for all43,455
shared zero-leg objects, despite matching magnitude and radius. It is not
used by this diagnostic.

The cached training eligibility is asymmetric. In case0 the zero-leg source
has43,455 physical detected rows, of which42,769 satisfy its stored
neighbour/distance rule; all43,452 sheared-leg source rows satisfy it.
Native nearest distances reproduce the stored distances in both legs.
The diagnostic preserves this original membership in T and BT, and uses
the fixed physical truth population for B/U/p. It does not silently change
the training population. Every case checks original NLL measurements and
contexts; cases overlapping detector preparation additionally check exact
float32 detector features and detection labels.

## Execution and review

Crowd-sharding16464026, case0 preparation16464027, and case0 GPU control
16464028 were submitted in sequence. Preparation array16464040 and GPU array16464041 cover the remaining159
cases, followed by review16464042. Preparation waits for successful case0 control. Each GPU array task then
waits for its corresponding preparation task (`aftercorr`); review waits
for the complete GPU array. The dependency update is recorded in
`submission.json`. Case0 is included once in the final160,
not an additional science case. Case0 preparation passed all86,221 cached
NLL rows with maximum context difference0 and exact detector feature/label
reproduction; it has48,460 truth objects,43,319 B pairs and42,633 BT pairs.
Case0 GPU control completed all16 groups with finite outputs, empty stderr,
and verified seed hashes (99,246,080 draws). The first four array preparations
(cases1,2,3,5) also passed with context error0 and empty stderr. Their
GPU tasks started with concurrency4 and completed their first independent
draw groups with finite outputs and empty stderr. Failed artifacts and frozen
worker versions are retained if an execution repair is necessary.

Local AST parsing, `bash -n` for all launchers, cut factorial/identity tests,
and the analytic forward projection sign/scale test pass. Heavy scans,
feature construction, sampling and full review run through Slurm. Follow
startup and first progress, then use a bounded one-shot check around2h.

The dependent review will verify all160 receipts, all16 groups, source and
output hashes, cached counts, detector reproduction, response identities,
and independently reconstructed pooled ratios/paired uncertainties.
Outputs will be `results.json`, `final_review.json`,
`review_sufficient.npz` and `completed.json`. Completion requires this
review, not just a successful scheduler state.

Neighbour shears average down under the primary projection; finite-scene
residuals and nonlinear effects at forward0.05 remain. Comparisons with
ConstGold also differ in shear stencil and neighbour coherence. Differences
between B, U and p are useful controls, not proof of two independent causes.

## Missing crowd matches and dependency repair — 2026-09-12

The owner observed `DependencyNeverSatisfied`. Original preparation array
16464040 completed153 tasks and failed6; including case0 there were154
successful preparations and20 completed GPU cases. The preparation worker
asserted that every physical truth row had finite crowd features, and stopped
an entire case when one lookup key was absent. These were preparation
failures, not memory exhaustion or nonfinite flow samples.

Probe16464414 verified one absent crowd lookup key in each failed case:

| Case | Truth input ID | Nonzero primary shear |
|---|---|---|
| 11 | 435292 | Yes |
| 26 | 64962 | No |
| 106 | 616611 | Yes |
| 142 | 72518 | No |
| 147 | 383340 | Yes |
| 189 | 232748 | No |

All six keys are also absent from both original training source catalogues
and the cached NLL row keys. Under the AGENTS matching rule, the repair
explicitly drops and reports these unmatched primaries. Three would already
be excluded by the zero-primary-shear rule; three are additional exclusions
from the intended response universe. No NLL rows or cached BT pairs are
lost. The full native scene remains available for neighbour features.
There is no imputation, invented crowd value, changed cut or model change.

Original workers, manifests, successful outputs, failed preparation
directories and logs remain intact. `repair1/manifest.json` freezes the exact
missing IDs and reused outputs, and `repair1/worker_hashes.json` freezes the
new workers. Any unexpected missing key or lost NLL row still fails. The
initial probe16464412 failed on the pandas keyword-only `axis` argument;
its source and log are preserved alongside the corrected probe.

Cancelled only pending original GPU tasks and their obsolete review.
Replacement preparation16464432 handles the six failed cases, then
gate16464433 checks all160 prepared receipts and data hashes, full NLL/BT
counts, and exactly six unmatched exclusions (three with nonzero shear).
GPU16464434 covers only140 unfinished cases, concurrency4, after the gate;
review16464435 combines these with the20 existing results. Replacement
dependencies use `afterok` after complete preparation verification.
The existing one-shot check follows only these four replacement jobs.

AST parsing, launcher `bash -n`, cut-factorial identities and forward
projection sign/scale self-tests passed before submission. All six repaired preparations completed with context error0 and empty
stderr. Gate16464433 completed successfully:160 verified preparations,
7,770,008 truth objects,6,943,102 B pairs,6,834,259 BT pairs and13,821,950
NLL rows. Its hashes/count checks confirm the exact six reported exclusions.
Replacement GPU16464434 started four tasks (cases11,26,29,30); each completed
first independent draw groups with finite outputs and empty stderr. Full
response results remain pending review16464435.

## Completed self-response results — 2026-09-12

All 160 science cases completed: 20 original GPU results were reused and 140
replacement results completed under 16464434. Summary/review 16464435 completed
exit0 in 6m40s with empty stderr. Independent completion review 16465903
completed exit0 in 1m42s, also with empty stderr. It checked all 320 case
receipts and 2,560 group hashes, reconstructed saved actual/model sums,
reproduced all 23-cut/five-parent responses and paired bootstrap/MC errors,
checked true-plus-residual identities, cut effects, Shapley allocations and
all saved subgroups. The original Slurm review also hashed native catalogues
and large prepared/per-object artifacts. No new sampling or fitting was done
for final verification.

The final population contains 7,770,008 nonzero-primary-shear truth objects,
6,943,102 matched usable B pairs and 6,834,259 cached BT pairs. All 13,821,950
NLL rows reproduced with maximum context difference **0**. Twelve cached
rows have zero primary shear and are excluded only from the response; their
cache reproduction remains checked. The six unmatched crowd primaries and
three additional response exclusions are listed in the repair section.

### Main m table and uncertainty

Percent m uses the [CONVENTIONS.md §8](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention)
ratio, with the same forward0→0.05 primary projection on both sides.
“No measured cuts” still retains the stated detection parent and physical
truth support. “No e cut” retains magnitude and measured-radius cuts.
Uncertainties below are paired case-bootstrap SE, conditional on the frozen
models; they do not include uncertainty from retraining.

| Population | No measured cuts: m (%) | No e cut: m (%) | All cuts: m (%) |
|---|---:|---:|---:|
| B: matched usable detections | +0.83205 ± 0.14903 | -1.21871 ± 0.13406 | -2.10987 ± 0.21772 |
| U: actual per-leg flags | +0.42780 ± 0.15030 | -1.50407 ± 0.13578 | -2.47962 ± 0.21835 |
| p: classifier probabilities | +1.50417 ± 0.15289 | -0.17124 ± 0.13622 | -1.09891 ± 0.22189 |
| BT: exact cached pairs | +0.74195 ± 0.14974 | -1.27748 ± 0.13577 | -2.17901 ± 0.21871 |
| T: exact cached rows | +0.43122 ± 0.15233 | -1.49680 ± 0.13974 | -2.44656 ± 0.21803 |

Separate integration SE in percentage points; these are much smaller than
the case uncertainty:

| Population | No cuts | No e cut | All cuts |
|---|---:|---:|---:|
| B | 0.000856 | 0.001779 | 0.004909 |
| U | 0.001040 | 0.001760 | 0.004771 |
| p | 0.000873 | 0.001788 | 0.004821 |
| BT | 0.000855 | 0.001743 | 0.004833 |
| T | 0.001019 | 0.001919 | 0.004756 |

### Measured and predicted parallel responses

The component here is primary-shear parallel self-response, not coherent
ConstGold R11. Both parallel/cross components and their uncertainties are
retained in `final_review.json`. All none/no-e/all cross-response residuals
for these five controls are below one paired case SE.

| Parent | Cuts | Measured R_parallel | Predicted R_parallel |
|---|---|---:|---:|
| B | none | +0.819858 ± 0.001209 | +0.813093 ± 0.000179 |
| B | no_e | +0.847868 ± 0.001163 | +0.858329 ± 0.000219 |
| B | all | +0.533669 ± 0.001278 | +0.545171 ± 0.000346 |
| U | none | +0.812502 ± 0.001221 | +0.809041 ± 0.000198 |
| U | no_e | +0.840874 ± 0.001163 | +0.853715 ± 0.000235 |
| U | all | +0.528813 ± 0.001267 | +0.542259 ± 0.000344 |
| p | none | +0.812502 ± 0.001221 | +0.800462 ± 0.000177 |
| p | no_e | +0.840874 ± 0.001163 | +0.842317 ± 0.000219 |
| p | all | +0.528813 ± 0.001267 | +0.534689 ± 0.000340 |
| BT | none | +0.815777 ± 0.001209 | +0.809769 ± 0.000181 |
| BT | no_e | +0.843448 ± 0.001172 | +0.854363 ± 0.000221 |
| BT | all | +0.531155 ± 0.001280 | +0.542987 ± 0.000345 |
| T | none | +0.811989 ± 0.001284 | +0.808502 ± 0.000352 |
| T | no_e | +0.840502 ± 0.001270 | +0.853273 ± 0.000377 |
| T | all | +0.528743 ± 0.001292 | +0.542003 ± 0.000403 |

### Detection controls and cancellation

The no-cut training result does **not** reproduce two negative half-percent
terms. U minus B changes m by -0.40425 ± 0.02497pp, but both no-cut controls
remain positive. The p minus U contrasts are:

| Cuts | m(p) - m(U), pp |
|---|---:|
| none | +1.07637 ± 0.01273 |
| no_e | +1.33283 ± 0.01329 |
| all | +1.38070 ± 0.01550 |

Thus the nearly zero p result without the e cut is an observed cancellation:
U gives -1.50407%, and changing to classifier weights shifts m +1.33283pp
to -0.17124%. With all cuts, -2.47962% plus +1.38070pp gives -1.09891%.
This control contrast includes changed population weighting and flow/detection
composition; it is not a uniquely identified standalone classifier error.
The sign differs from the coherent ConstGold control.

The cached eligibility asymmetry does not explain the large selected-response
mismatch here: T minus U is +0.00342 ± 0.02312pp without cuts and
+0.03306 ± 0.03163pp with all cuts. BT minus B is smaller but resolved:
-0.09010 ± 0.02363pp without cuts and -0.06914 ± 0.02280pp with all cuts.
The exact cached pairs still show the same sign change under selection.

### Cut factorial, drop-one effects and interactions

The measured-radius cut has the largest effect in this training self-response
test. With actual flags, shape alone gives +0.11911 ± 0.24704%, radius
alone gives -1.50171 ± 0.13692%, and both give -2.47365 ± 0.21534%.
The magnitude cut adds little at the preset threshold.

| Measured cuts | B m (%) | U m (%) | p m (%) |
|---|---:|---:|---:|
| none | +0.83205 ± 0.14903 | +0.42780 ± 0.15030 | +1.50417 ± 0.15289 |
| shape | +0.59743 ± 0.24967 | +0.11911 ± 0.24704 | +1.28538 ± 0.25088 |
| magnitude | +0.74834 ± 0.15074 | +0.38423 ± 0.15207 | +1.48439 ± 0.15442 |
| shape_magnitude | +0.58447 ± 0.25961 | +0.13719 ± 0.25603 | +1.32671 ± 0.25991 |
| radius | -1.21874 ± 0.13553 | -1.50171 ± 0.13692 | -0.17595 ± 0.13742 |
| shape_radius | -2.11032 ± 0.21470 | -2.47365 ± 0.21534 | -1.09735 ± 0.21909 |
| magnitude_radius | -1.21871 ± 0.13406 | -1.50407 ± 0.13578 | -0.17124 ± 0.13622 |
| all | -2.10987 ± 0.21772 | -2.47962 ± 0.21835 | -1.09891 ± 0.22189 |

Adding the named cut to the other two cuts gives the following paired
change; effects do not add because the selection regions interact:

| Parent | Added cut | Change in m, pp | Change in R_pred - R_measured |
|---|---|---:|---:|
| B | shape | -0.89116 ± 0.21994 | +0.001042 ± 0.001391 |
| B | magnitude | +0.00045 ± 0.03224 | -0.000002 ± 0.000176 |
| B | radius | -2.69434 ± 0.21207 | +0.014438 ± 0.001095 |
| U | shape | -0.97555 ± 0.22017 | +0.000605 ± 0.001388 |
| U | magnitude | -0.00597 ± 0.03340 | +0.000035 ± 0.000181 |
| U | radius | -2.61681 ± 0.21325 | +0.014132 ± 0.001098 |
| p | shape | -0.92767 ± 0.22392 | +0.004433 ± 0.001388 |
| p | magnitude | -0.00157 ± 0.03385 | +0.000009 ± 0.000181 |
| p | radius | -2.42563 ± 0.21583 | +0.012429 ± 0.001096 |

For U, adding the e cut changes m by -0.97555 ± 0.22017pp, but the
absolute response error changes only +0.000605 ± 0.001388: much of the
change in relative m comes from the smaller response denominator
(0.853715 to 0.542259). The shape-radius interaction with magnitude fixed
is -0.72851 ± 0.18310pp in m, but -0.001846 ± 0.001073 in absolute
response error. For p the corresponding values are -0.77000 ± 0.18530pp
and -0.001000 ± 0.001073. A strong interaction in a ratio is not by itself
evidence for an equally strong interaction in the underlying response error.

The order-averaged Shapley allocation of the change in absolute response
error from no cuts to all cuts is below. This is an accounting convention,
not a unique causal attribution:

| Parent | Shape allocation | Magnitude allocation | Radius allocation |
|---|---:|---:|---:|
| B | +0.002316 ± 0.001305 | +0.000220 ± 0.000152 | +0.015731 ± 0.000962 |
| U | +0.001669 ± 0.001306 | +0.000112 ± 0.000154 | +0.015127 ± 0.000961 |
| p | +0.005019 ± 0.001309 | -0.000023 ± 0.000153 | +0.012920 ± 0.000960 |

### True and measurement-residual components

With the same selected denominator on each side, the exact identity is
`R_selected_measured = R_selected_true + R_selected_(measured-true)`.
The table reports model minus data for each component, with paired case SE:

| Parent | Cuts | Delta R_true | Delta R_(measured-true) | Total delta R |
|---|---|---:|---:|---:|
| B | no_e | +0.001743 ± 0.000409 | +0.008717 ± 0.001063 | +0.010461 ± 0.001151 |
| B | all | -0.004499 ± 0.000902 | +0.016001 ± 0.001077 | +0.011502 ± 0.001186 |
| U | no_e | +0.000215 ± 0.000415 | +0.012626 ± 0.001072 | +0.012840 ± 0.001159 |
| U | all | -0.005556 ± 0.000907 | +0.019001 ± 0.001108 | +0.013446 ± 0.001183 |
| p | no_e | -0.000178 ± 0.000464 | +0.001620 ± 0.001075 | +0.001442 ± 0.001147 |
| p | all | -0.002846 ± 0.000941 | +0.008722 ± 0.001125 | +0.005876 ± 0.001186 |

For U without the e cut, the true-shape term is consistent with zero;
almost all of the +0.012840 response error is in measured-minus-true.
With all cuts, -0.005556 and +0.019001 partly cancel. For p with all cuts,
-0.002846 and +0.008722 also partly cancel. The selected measurement
distribution remains mismatched even on training scenes without R_blend.

### Preset thresholds and subgroup limits

Other cuts stay at their preset values. These are diagnostic curves; no
threshold was chosen or optimized from these results.

| Varied cut | Threshold | B m (%) | U m (%) | p m (%) |
|---|---:|---:|---:|---:|
| shape | 0.5 | -1.700 ± 0.290 | -2.087 ± 0.294 | -0.661 ± 0.298 |
| shape | 0.55 | -2.249 ± 0.274 | -2.632 ± 0.276 | -1.237 ± 0.281 |
| shape | 0.6 | -2.110 ± 0.218 | -2.480 ± 0.218 | -1.099 ± 0.222 |
| shape | 0.65 | -2.223 ± 0.216 | -2.580 ± 0.222 | -1.212 ± 0.227 |
| shape | 0.7 | -2.289 ± 0.196 | -2.631 ± 0.198 | -1.273 ± 0.201 |
| magnitude | 25.3 | -2.362 ± 0.237 | -2.738 ± 0.237 | -1.310 ± 0.242 |
| magnitude | 25.5 | -2.231 ± 0.227 | -2.600 ± 0.229 | -1.201 ± 0.232 |
| magnitude | 25.8 | -2.110 ± 0.218 | -2.480 ± 0.218 | -1.099 ± 0.222 |
| magnitude | 26.0 | -2.087 ± 0.217 | -2.456 ± 0.218 | -1.078 ± 0.222 |
| magnitude | 26.3 | -2.111 ± 0.215 | -2.476 ± 0.215 | -1.099 ± 0.219 |
| radius_arcsec | 0.6 | +0.488 ± 0.263 | +0.128 ± 0.259 | +1.317 ± 0.263 |
| radius_arcsec | 0.675 | -0.899 ± 0.229 | -1.249 ± 0.228 | -0.011 ± 0.232 |
| radius_arcsec | 0.75 | -2.110 ± 0.218 | -2.480 ± 0.218 | -1.099 ± 0.222 |
| radius_arcsec | 0.825 | -3.627 ± 0.247 | -4.069 ± 0.249 | -2.443 ± 0.253 |
| radius_arcsec | 0.9 | -5.319 ± 0.291 | -5.818 ± 0.292 | -3.851 ± 0.298 |

In particular, increasing the radius threshold from0.60 to0.90arcsec moves
U m from +0.128% to -5.818%, and p m from +1.317% to -3.851%. This is
strong sensitivity to radius selection, despite response regularization
on the population mean. It points to the selected joint distribution; it
does not identify one unique physical or statistical cause.

Saved subgroups use B with all cuts, so they are not direct decompositions
of the p detector contrast. They are overlapping marginal descriptions,
not mutually adjusted causal effects. Selected counts differ by leg.
Faint true-magnitude bins24–25 and25–25.8 give m=-4.458±0.524% and
-4.047±0.699%; true semi-major Re1.0–1.5 gives -6.573±0.548%.
Near/above the e-cut boundary it is safer to report R than relative m:

| Intrinsic e magnitude | Measured R_parallel | Predicted R_parallel | Selected data rows: sheared / zero |
|---|---:|---:|---:|
| 0.4–0.6 | +0.087398 ± 0.003033 | +0.111886 ± 0.000804 | 1,094,069 / 1,100,743 |
| 0.6–0.8 | -0.016419 ± 0.008548 | +0.051304 ± 0.000888 | 150,624 / 150,515 |
| 0.8–1.0 | +0.050195 ± 0.069012 | +0.168363 ± 0.003165 | 2,655 / 2,658 |

The highest-e bin is sparse and noisy; large subgroup ratios are not a
calibration statement for the whole population. Full subgroup sums and
uncertainties remain in `completion_review.json` and `results.json`.

### Tail checks, scope and completion

All15,912,976,384 draws passed the worker finite-output checks. Numerical
zero radii8,473 and zero fluxes33 were retained in the no-cut populations
and handled by the fixed cut comparisons. Maximum sampled radius was
10,421.047pixels and flux87,247,756.53 in the bundle units; these finite
tails were retained rather than clipped. No claims about untested tail
probabilities follow from finite integration SE.

This is an in-sample flow-scene test. Nineteen cases overlap detector
training and four overlap detector tuning; no model was selected by these
results. The forward0.05 random-direction self-response differs from
ConstGold central0.02 coherent response, and is not actual SBSI inference m.
The results establish that a selection-response mismatch exists without
R_blend in training scenes; they do not assign a numerical share of the
ConstGold residual to one mechanism. Accurate mean radius response alone
does not guarantee accurate shape response after a measured-radius cut.

Both one-shot watches are disabled and only this experiment’s tagged cron
entry `sbsi-training-self-selection-once` is removed. No training, fitted
correction, threshold change, model promotion or inference was performed.

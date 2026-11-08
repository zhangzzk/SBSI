# Review of recent detector experiments — 2026-09-13

## Verdict and scope

The recent experiments are reproducible development work. The checked code
implements the declared BCE objectives, case splits, frozen base models and
equal probability ensembles. I found no empirical response offset, scaling,
response-selected checkpoint or seed weighting in the three coherent-U
workers. The latest candidate does not establish 0.3% calibration, and keeping
it out of the configured likelihood is justified.

The stronger conclusion that detector-only factorization has reached its
scientific limit is not established. The conditioning argument omits the
external blend-response shift, the final fit remains limited by its epoch
budget, and the response screen is not independent of the full model's
development. A stopping decision on this exhausted endpoint is reasonable;
it should not be described as proof that a model family cannot work.

Scope: WORKLOG cont.623–626 and the usable-label correction in cont.619,
their external workers/manifests/training logs/result receipts, the inherited
coherent data and flow-moment builders, and the relevant conventions. This is
not an exhaustive review of every uncommitted experimental file in SBSI.
No scientific models, thresholds, experiment records or release settings were
changed during this review.

## Finding 1 — Include R_blend in the conditioning argument

Priority: P1 for the interpretation and next-experiment design.

[The status argument](CALIBRATION_STATUS.md#where-the-no-e-error-lives) and
[the eight-feature interpretation](DETECTION_CIRCULARIZED_REPAIR.md#coherent-u-flow-condition-follow-on--2026-09-13)
use the tower identity to argue that nesting the eight flow conditions is
sufficient for the evaluated deterministic moment. The identity is correct
for a moment measurable in those conditions. The actual complete moment has
an additional input.

The inherited `selection_replication_c90_139_v1/run.py:133` constructs
`shift = R_blend * (es - e0)`. The inherited
`selection_localization_v1/run.py:200` adds this shift to every sampled shape
before computing the selected numerators. `R_blend` is a separately supplied
sum of neighbour-emulator predictions, not one of the eight flow inputs.

For no-|e| cuts, write the flow acceptance probability as `q(F_g)` and its
unshifted selected first moment as `u_flow(F_g)`. The evaluated numerator is

```text
u_g = u_flow(F_g) + q(F_g) R_blend (e_g - e_0).
```

Thus matching `P(U|F_g)` exactly does not by itself guarantee
`E[p_g u_g] = E[U_g u_g]`. The detector's conditioning information must also
make the blend term measurable, or an appropriate conditional covariance
must vanish. With an ellipticity cut, the shift can affect `q` as well. These
identities must hold under each shear-leg distribution, not merely after
pooling the training legs.

A finite counterexample makes the missing assumption explicit: let `F` be
constant and `U=R_blend=Z`, where Z is equally likely to be zero or one. The
perfect detector on F returns 1/2, yet `E[p R_blend]=1/4` and
`E[U R_blend]=1/2`.

Correct the interpretation to distinguish the unshifted flow moment from the
complete moment. This does not invalidate the computed responses, nor prove
that the omitted covariance causes the measured discrepancy. It does mean
the eight-feature failure cannot be attributed solely to finite-network BCE
accuracy using the stated argument. A future joint model must also state
whether it absorbs the external neighbour response or retains it explicitly.

## Finding 2 — Distinguish a stopping decision from a model-family limit

Priority: P2.

[The final disposition](DETECTION_CIRCULARIZED_REPAIR.md#coherent-u-full-scene-residual-follow-on--2026-09-13)
says that component failure shows detector-only factorization has reached its
limit. It tests one finite set representation, optimizer recipe and training
budget on a repeatedly reused endpoint. That result supports rejecting this
candidate under the frozen gate, not excluding every detector-only model.

The same section calls job16468939 “converged.” All three tasks stop at
epoch100, selecting100/99/99. None satisfies the ten-epoch patience stopping
rule. Tune BCE between epochs80 and100 continues to decrease by about
0.0000509, 0.0000422 and 0.0000942. These gains are small, and do not show
that further training would improve response, but the histories do not
establish convergence at the required response precision. The earlier
30-to-100 extension was recorded using tune BCE before test prediction;
that is a defensible development adjustment.

The reported truth/residual cancellation is numerically real. However, their
negative correlation is not independent evidence of a mechanism: the
residual is defined by subtracting truth from the full moment. The
non-worsening component gate is a prespecified robustness requirement,
stronger than the condition that the total response be accurate. Failing it
does not imply fabricated or mathematically undefined results.

Describe the outcome as “this bounded detector experiment failed its
development gate; further tuning on this endpoint is retired.” Treat the
shared-scene measurement model as a motivated hypothesis for fresh data.
Do not retrospectively extend training after seeing this response screen.

## Finding 3 — The inherited 69.2% gate no longer enforces central 0.3%

Priority: P2 for future gate reuse; the present rejection is unchanged.

The original100-case raw-D crowd worker includes central `abs(m)<=0.3` in
its pass boolean. The newer coherent-U workers retain the69.2% composition
reduction but record central0.3% entry only as a descriptive boolean. For
example, the full-scene `worker.py:606–618` does not include that boolean in
`passes_prespecified_mechanism_screen`.

After reducing the screen to cases120–139, the U-oracle response is
1.0084148448 and the measured response is1.0059616462. Keeping their observed
signs, the allowed positive candidate-minus-U response is0.0005737672. The
current detector's corresponding gap is0.0059661052. Central entry therefore
requires about90.383% reduction, rather than69.2%. Exactly69.2% would give
`m=-0.42472%`; it can also satisfy both component non-worsening conditions.

The newer manifests explicitly describe a mechanism screen, so this does
not violate their literal frozen rule. It does break equivalence with the
original target-based gate. Preserve the published decisions and either
state that the gate is only a heuristic mechanism screen or restore an
explicit central-target condition in the next prospectively frozen protocol.
The full-scene candidate still fails its current truth-component gate.

## Finding 4 — Make the holdout and catalogue-reuse boundary explicit

Priority: P2 for scientific reporting.

Cases120–139 are held out from fitting these new detectors, but are not a
holdout for the complete measurement model. The lambda10 training lineage
uses `response_target_mixed_train160_6x6x5_matrix_re037.npz`:14 of those20
cases are in its training split and the other6 in validation. These are
different shear products, not evidence that these exact coherent image
measurements entered training; they still share the scene-case universe.
Architecture and sample choices also repeatedly used this response endpoint.

The current experiment manifests acknowledge endpoint reuse. The older
[catalogue contract](CONVENTIONS.md#2a-constgold--the-constant-shear-antithetic-catalogue)
still declares cases40–139 evaluation-only and prohibits training on
ConstGold. New workers train on fully targeted coherent raw catalogues from
cases40–99 and tune on100–119. Document the deliberate reclassification and
distinguish these raw catalogues from the historical both-detected response
table. Do not describe this split as untouched full-model validation.

The published intervals resample20 cases conditional on the fitted models
and already averaged16 QMC groups. They do not include a separate candidate
QMC integration estimate, measurement-flow training-seed uncertainty, or
adaptive model-selection uncertainty. That is adequate for a labelled
development diagnostic. It is not an unconditional certification interval.
The candidate residual-component interval only narrowly excludes zero, so
its significance should retain this qualification.

## Reproduced evidence

Every row below uses cases120–139, central coherent g1±0.02, per-leg usable
selection, magnitude/radius cuts with no measured-|e| cut, the frozen lambda10
epoch154 flow, trial9 external blend response, and the saved mean of16×64
CRN QMC draws. `m=100(R_actual/R_model-1)`; uncertainties are case-bootstrap
standard errors in percent. See [response and matching conventions](CONVENTIONS.md#6-response-estimators).

| Detector | m (%) | Case SE (%) | 95% case interval (%) |
|---|---:|---:|---:|
| Coherent U, eight flow conditions | -0.777910 | 0.221964 | [-1.206143, -0.337346] |
| Coherent U, rich fixed vector | -0.663783 | 0.222300 | [-1.090862, -0.223632] |
| Coherent U, full-scene residual | -0.279359 | 0.223532 | [-0.709075, +0.164826] |

For the final candidate, candidate-minus-U truth/residual response is
`+0.0010826885 - 0.0007177711 = +0.0003649174`. Their95% intervals are
`[0.0002647777,0.0019288862]` and
`[-0.0014747686,-0.0000153529]`. The total composition reduction is93.8835%.
These reproduce the work log; the interval for m fails the0.3% acceptance rule.

The audit independently pooled saved sufficient statistics and used explicit
repeated-case indexing for10,000 bootstrap draws, without importing either
experiment reducer. All three ensemble point/SE/interval results agree within
1e-11. The12/19/18 checked source/input/output hash mappings and all60
individual case-result hashes and arrays match. Aggregate full=truth+residual
identities agree within4.36e-15.

Slurm raw-source audit16469851 completed in12s, exit0, empty stderr. Cases40,
100 and120 reproduce both saved U masks exactly from the fully targeted
crossmatch/shape catalogues. All ten checked truth columns agree exactly
between the scene-builder and coherent input files; stored intrinsic shape
agrees with the axis-ratio/angle construction within8.68e-16. No crossmatch
lacks a shape row in those six legs; unmatched shape rows are counted and
dropped. No retained usable shape has a g2 sentinel or lies outside the unit
disk. This checks three cases, not the entire raw100-case input collection.

The scene-worker self-test passes. Additional tiny checks with a nonzero
correction pass neighbour permutation invariance, padding-mask invariance,
zero-neighbour finiteness and frozen-base gradients. The imported selection
and shear helpers have exactly the frozen source hashes. Scheduler records
confirm final training16468939, case array16469024 and reviews16469047/49
completed successfully; only the preserved initial setup failure has nonempty
stderr in the final experiment directory.

## Audit artifacts and validation boundary

Independent audit scripts and JSON outputs are retained under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/methodology_review_20260913_v1/`:
`numerical_audit.py`, `numerical_audit.json`, `raw_audit.py`, `raw_audit.json`,
and the raw Slurm stdout/stderr. The numerical audit runs with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and the documented py31 interpreter;
the raw-catalogue audit was run through Slurm on the inter partition with2CPUs
and8GB RAM.

This review did not retrain models, generate new flow draws, inspect the
sealed140–239 candidate endpoint, or run full posterior inference. The
release configuration and all existing scientific artifacts remain unchanged.
Documentation whitespace validation uses `git diff --check`; a full package
test run is not required for these documentation-only repository additions.

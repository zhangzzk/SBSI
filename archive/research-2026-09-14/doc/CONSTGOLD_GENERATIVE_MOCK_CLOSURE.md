# ConstGold generative mock catalogue closure

## Recipe and interpretation — 2026-09-12

The owner proposes drawing measurements from the sheared likelihood, applying
the blend shift, sampling detection flags and cutting the mock measurements,
then comparing their response and offsets with the image measurements. This
diagnostic uses the same finite truth population and models as
[selection localization](CONSTGOLD_SELECTION_LOCALIZATION.md#completed-full50-review--2026-09-12).
Response extraction and sign follow [CONVENTIONS §6](CONVENTIONS.md#6-response-estimators)
and [§8](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).

Artifacts are under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/generative_mock_closure_v1`.
The frozen `manifest.json`, `workers.json`, `run.py` and `submission.json`
record the recipe, dependencies and scheduled jobs. Array 16460504 evaluates
50 cases with four concurrent V100 workers; dependent review 16460505 checks
all receipts and produces `results.json`.

- Cases 40–89: 4,854,422 fixed truth objects, strict true magnitude
  `18<r<25.8` and semi-major radius `0.37<Re<1.5 arcsec`.
  Complete neighbours and circularized flow conditioning are retained.
- Lambda10 shape+radius flow epoch154, ordinary-MSE trial9 blend prediction,
  and circularized-BCE detection probabilities. Prepared frames, blend shifts
  and detector probabilities are reused with hash and object-order checks.
- 64 independent full catalogue realizations. Each object has one ordinary
  Gaussian-latent flow measurement and one Bernoulli detection flag per
  realization. A fixed object can be detected zero or one times; Poisson
  counts would allow repeated detections of that object.
- The blend shift is added before cuts. Report no measured cuts, magnitude
  plus radius cuts, and all cuts: `MAG_AUTO<25.8`, radius at least 3.75 pixels,
  and, for the last configuration, `|e|<0.6`.
- Separate random streams for flow and detection, distinct between cases;
  shared latent noise and detection uniforms across the two shear legs and
  shared draws across cut configurations. This coupling reduces numerical
  noise; it is not a claim about physical cross-leg detection transitions.
- Save the complete first mock realization for each case, including rejected
  candidates, physical measurements, truth IDs, flags, probabilities and
  blend shifts. Save exact per-case moments for all 64 realizations.

For each leg, the current numerical prediction estimates

\[
\bar e_g^{\rm pred}=
\frac{\sum_i p_i(g)\,\mathbb E[e_iS_i\mid x_i(g),D_i=1]}
     {\sum_i p_i(g)\,\mathbb E[S_i\mid x_i(g),D_i=1]}.
\]

Thus the old prediction already integrates mock measurements after their own
cuts. The explicit catalogue adds Bernoulli realization of detection and
uses one measurement per object. It tests the same marginal generative model.
The mean of a finite-catalogue ratio can differ from the ratio of expected
moments; both pooled-moment estimates and the distribution of individual
catalogue ratios are retained. Probability-weighted controls on exactly the
same new measurement draws isolate the effect of sampling detection flags.
The previous 16-group, 64-draw QMC prediction supplies an independent
numerical reference.

The generative factorization is
`P(D=1|x,g) P(y|D=1,x,g)`. Drawing a flag and a detected-conditional
measurement with separate random streams given the conditions implements
this factorization; it does not assume unconditional detection–measurement
independence. Actual measurements retain the audited usable-detection masks
and catalogue-matching exclusions of the preceding investigation.

## Reported quantities and limitations

With `g1=+/-0.02`, `g2=0`, pool selected moments over cases before forming means:

\[
R_{i1}=\frac{\bar e_{i,+}-\bar e_{i,-}}{0.04},\qquad
a_i=\frac{\bar e_{i,+}+\bar e_{i,-}}{2}.
\]

The actual-versus-mock response closure is
`m=R_actual,11/R_mock,11-1`. The independent self-consistency test is
`R_mock,11/R_reference,11-1`. These are separate questions: a model can pass
its own numerical closure and disagree with the image measurements.

The reported `c1` proxy is `(a_actual,1-a_mock,1)/R_mock,11`, a shear-equivalent
offset difference. There are no zero-shear image legs in these cases, so this
is a finite-shear midpoint diagnostic, potentially including even nonlinear
terms; it is not a separate zero-shear additive-bias measurement. We retain
raw offsets in both shape components. No g2 shear pair is available to
measure R22 or m2. None of these quantities is actual SBSI inference m/c.

Uncertainties use 10,000 paired case bootstrap draws, seed20260902, plus
separate delete-one-replica integration errors. The previous 16-group
reference integration uncertainty is reported separately. Case errors
already contain finite integration noise and should not simply be added in
quadrature to that component. Independent mock realizations are not new
independent image simulations.

Validation includes analytic synthetic response/sign and Bernoulli moment
checks, independent retained-row reconstruction of every persisted mock,
reconstruction of actual moments from the prepared data, source/output hashes,
object counts, cut boundaries and exact measured=true+residual identities.
No training, threshold fitting, correction or model promotion is part of this
diagnostic. Previous completed watches remain disabled.

## Completed comparison — 2026-09-12

All 50 array tasks16460504 completed with exit0, taking16–38 seconds each;
review16460505 completed43 seconds with exit0. Every stderr file is empty.
The review verified all source/prepared/output hashes and 50 completed
receipts, reconstructed the first mock's statistics directly from retained
rows, reproduced actual catalogue sums, and checked exact component
identities. Independent `check_results.py` also reconstructs responses and
offset proxies from compact sums and checks integration SE against the
independent catalogue spread.

All 621,366,016 draws were finite. There were349 numerical zero radii and
2 zero fluxes, retained under the existing cut semantics; the largest radius
was6973.0139 pixels and largest flux1.0449690e7 in ZP30 units.

| Measured cuts | Actual R11 | Explicit mock R11 | Previous numerical R11 | Actual/mock m, percent ± case SE |
|---|---:|---:|---:|---:|
| None | 0.95029592 | 0.96114702 | 0.96114671 | -1.12897 ± 0.22452 |
| Magnitude + radius, no e cut | 1.00459892 | 1.01400212 | 1.01407133 | -0.92733 ± 0.17505 |
| All three | 0.60537833 | 0.61797614 | 0.61816017 | -2.03856 ± 0.30594 |

Actual R11 case SE are0.00215464/0.00175045/0.00188801, in table order;
mock case SE0.00026970/0.00027302/0.00045435. Separate mock integration SE
on R11 are0.00002864/0.00008580/0.00017903 and on actual/mock m are
0.00295/0.00838/0.02838 percentage points. Case and MC components are not
added as independent errors.

Mock/reference self-closure m values are+0.000032/-0.006826/-0.029771 percent,
with mock MC SE0.002980/0.008461/0.028961pp and independent reference MC
SE0.000516/0.001328/0.004810pp. All agree within their sampling uncertainty.
The mean of64 individual catalogue R11 ratios minus the response from pooled
moments is only+1.72e-9/+1.92e-8/-7.61e-9. Finite-ratio averaging does not
explain the percent-level discrepancy here.

On identical measurement draws, Bernoulli minus probability-weighted R11 is
-0.00000280/+0.00001086/+0.00002486, with paired MC SE
0.00002702/0.00003468/0.00005114. Explicit detection realization therefore
reproduces the previously integrated probability-weighted prediction.

The mocks also disagree with the image catalogues in their even mean-shape
offsets. Values below are **actual minus mock**, in units of1e-4, with paired
case SE. `c1` is the finite-shear offset-difference proxy defined above.

| Cuts | Delta a1 × 1e4 | Delta a2 × 1e4 | c1 proxy × 1e4 |
|---|---:|---:|---:|
| None | +11.0527 ± 1.4714 | -23.0989 ± 1.4902 | +11.4995 ± 1.5301 |
| Magnitude + radius | +7.6248 ± 1.6767 | -15.5247 ± 1.8796 | +7.5195 ± 1.6531 |
| All three | +5.7991 ± 1.3421 | -11.0269 ± 1.5328 | +9.3840 ± 2.1730 |

For no e cut, actual `(a1,a2)=(+0.00016027,-0.00004034)`, while the mock
gives`(-0.00060221,+0.00151213)`. This discrepancy concerns the mean itself,
in addition to its shear derivative. It is not yet an identified physical
cause or an inference-bias estimate. The absence of g=0/g2 image legs still
limits the additive interpretation; no c2 calibration is inferred using R11.

The generative test supports the numerical response calculation and confirms
a discrepancy between these model-generated catalogues and the image
measurements. Shared prepared conditions and blend predictions mean this is
not an independent audit of every condition-construction assumption. It
does not isolate a unique detector, flow, blend or missing-context cause.
No correction or model change was applied. `results.json`,
`replica_metrics.json`, `sufficient.npz`, `independent_check.json` and
`completion_review.json` retain the full results and provenance.

# Selected-neighbour moment experiment

## Training and held-out selection

Both fixed step64 endpoints completed. The zero-weight control reproduces
the previous neighbour-conditioned nine-cut checkpoint exactly on1024
log-density checks and its500k-row validation NLL. The new auxiliary moment
is a training constraint, with no correction applied to predictions.
See [the frozen design](rule.md#additional-objective).

At radius>3 pixels and magnitude<25, the original unsheared usable rows give
the following measured-minus-generated selected reference-neighbour
coefficient errors. These are **not shear-response errors**. Uncertainties
are whole-case bootstrap SEs with fixed models and fixed sampled rows.

| Population | Nine-cut control | Nine-cut plus neighbour moment |
|---|---:|---:|
| Training g0 | +0.003322 ±0.000424 | +0.001024 ±0.000424 |
| Validation g0 | +0.003148 ±0.000519 | +0.000897 ±0.000529 |

The paired validation change is -0.002251 ±0.000026. Training and validation
improve by69% and72%, respectively. The pooled original g0/g05 validation
error changes from+0.002823 to+0.000544. Soft-cut diagnostics improve in the
same direction; the result therefore transfers from the smooth training
objective to hard analysis cuts. Validation NLL changes from-3.914690 to
-3.918241 (lower is better). The starting NLL-only checkpoint had-3.921348,
so this recovers only part of the likelihood cost of response regularization.

Original response probes show a tradeoff, not a uniform improvement. In
training, model-minus-simulation R11 changes from+0.004934 to+0.003126 while
R22 changes from-0.001013 to-0.004691. These are the fixed stochastic training
probes, without an attached case uncertainty, not constant-shear calibration
results. All four response components remain in the penalty.

## Joint constant-shear evaluation

All four scenes and both axes are now complete; all jobs exit0:0. At the
target cut with actual usability and generated selection, m1 changes from
+1.0564% ±0.6041pp to+1.1300% ±0.6114pp, and m2 changes from+1.3794% ±0.4184pp
to+1.7015% ±0.4219pp (case-bootstrap SE). Paired shifts are+0.07358 ±0.00986pp
and+0.32209 ±0.00854pp; corresponding jackknife SEs are0.01128/0.00991pp.
All nine cuts shift upward on both axes. These four reused development
scenes do not provide independent acceptance, and small paired SEs do not
include uncertainty in fitted models.

In absolute response units, the added constraint changes the model flow
contribution by-0.003246 in R11 and-0.005219 in R22, while the emulator
contribution rises by+0.002594/+0.002379. Thus the positive residual increases
by+0.000652/+0.002840. Replacing generated selection with measured selection
still gives residual shifts+0.000734/+0.002969 while retaining generated
photometry. The reference-neighbour score error on the common-four-usable
constant-shear cohort falls from+0.003112 to+0.000866: the proxy improvement
transfers to these scenes, but does not ensure total response calibration.

The full evaluator reproduces the previous control results. All numerical
receipts pass, and the maximum pooled/separate emulator discrepancy is
8.88e-16. Held-out NLL improves by-0.003551 ±0.000221. Own-data half-bank
score differences at the target are below0.000062, well below the roughly
0.00225 paired improvement. See [full results](results.md) and `report.json`.

### Earlier pilot

The case1000 axis1 pilot completed in378 seconds for both endpoints. Pooled
emulator calls agree with separate calls to4.44e-16; measured-selection mass,
reference-score replay and simulation sums pass. In this one scene, actual
usability plus generated selection gives m1=+0.4722% for the control and
+0.5608% for the added-moment endpoint. The neighbour contribution rises by
0.002610 while the flow contribution falls by0.003398, leaving a net model
response reduction of0.000788. Predicted-usability m1 similarly changes from
+0.2698% to+0.3421%. Thus the proxy improvement does not by itself improve
the first scene's total calibration. There is no between-case SE for this
single-scene check; these are contribution changes, not standalone errors.
Numbers are retained in `pilot_summary.json`.

The three
flag/selection modes still use generated emulator photometry. The separate
full measured-photometry intervention is documented in
[its forty-scene findings](../joint_measured_photometry_20261106/findings.md#full-forty-scene-result).

The older broad evaluation remains held. No endpoint is promoted and no
constant-shear outcome chooses the loss weight or endpoint.

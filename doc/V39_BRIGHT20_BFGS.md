# V3.9 five bright-only BFGS iterations

## Completed five updates on September 27

All four workers16721029[0–3] completed successfully in1h37m40s or less.
The fifth and final estimate is g1=0.0205598964 ±0.0003832200 and
g2=0.0009044658 ±0.0003184539 (conditional row SE). Its step is
(3.14e-7,2.88e-6), giving two consecutive small steps: iteration stability
passes. Relative g1 realization offset is+2.799% ±1.916 percentage points.

The paired8192-to16384-draw difference is(-0.0001270075,-0.0000891862),
with conditional SE(0.0000858462,0.0000830116). Draw stability still fails;
no0.3% acceptance or validation of the faint-carry approximation follows.
The owner requested a [matching actual-measurement run](V39_ACTUAL20K_BFGS.md#request-and-sample)
after this experiment. The mock's frozen driver and results are preserved.

## Progress at 00:24 CEST on September 27

Four of five bright updates are complete; all four workers remain running
without failures. The fifth update has completed12 of20 normalization shards,
with the next four shards in progress. Recent iterations took about24minutes;
approximately10–15minutes remain at the observed rate.

| Bright iteration | g1 | g2 | Conditional SE(g1) |
| --- | ---: | ---: | ---: |
|1, reused matching saved evaluations |0.02076256 |0.00075840 |0.00068031 |
|2 |0.02047071 |0.00092421 |0.00056625 |
|3 |0.02056567 |0.00091328 |0.00042537 |
|4 |0.02055958 |0.00090159 |0.00037764 |

Iteration4 moved g1 by-6.09e-6 and g2 by-1.17e-5, satisfying the small-step
screen once. Two consecutive passes are required, so formal iteration
stability is not yet established. Relative to injected g1=0.02, the current
offset is+2.798% ±1.888 percentage points (conditional row SE). The g2
conditional SE is0.00041802. These are intermediate realization offsets,
not repeated-realization bias measurements.

The paired8192-to16384-draw g1 shift remains-0.00012883 ±0.00008480,
or-0.644% ±0.424 percentage points relative to0.02. It fails the declared
draw-stability screen. A stable fifth iteration would not by itself validate
the measured-mag20 carry or establish0.3% accuracy.

## Owner instruction and starting point

On September26 the owner requested five iterations recomputing only bright
observations with magnitude<20, then clarified that the chain must start
from the first full pass, not from the later proposed BFGS centre0.019166.
The implemented start is the original first full-pass Gaussian estimate
`(0.025084524129982133, 0.0026760181941245837)`. The BFGS matrix begins as
the original full pass's observed information; no information from the later
full-pass BFGS proposal initializes the solver.

The fixed reference for carried observations is that original full evaluation
at raw mean shape `(0.013740481623385736, -0.002647580061745958)`.
Every later faint-row carry is computed once from this same reference.
The magnitude threshold uses measured flux via `30-2.5*log10(flux)`, strictly
less than20. It selects142 of20000 observations. All20000 remain in the
likelihood estimate; this is an evaluation shortcut, not a population cut.
The model-generated sample still has measured radius>0.6arcsec and magnitude
<25.8, without truth-property cuts.

## Frozen numerical procedure

Keep F8/C8/original physical-moment disk models, the pre-extension20,231,221
weighted prior, Gaussian reranking of131072 location candidates to K1024,
tilted-stratified delta0.1, proposal seed8701, h0.001, and nested512–16384
draws. Do not mix the whole-prior-ranking probe with these saved full-pass
moments. The primary solver uses the16384 rung and the historical positive
curvature BFGS secant update; the other rungs use that same solver matrix
only for conditional paired draw diagnostics.

Carry numerator score/information linearly and replace the common log
selection-normalization derivatives on **every** observation. Recompute
the full weighted-prior normalization at every new centre, with64 common
random-number QMC samples per atom and the same per-shard seeds from8101.
Record observed information separately from the BFGS solver matrix.

Preserve each bright object's original four-observation density group.
This evaluates560 rows including418 padding rows, but only142 measured-bright
rows replace carried moments. Padding-row results are discarded. GPU replay
requires all ten retained moment/diagnostic arrays to equal their saved
full-pass values exactly on those560 rows.

Run five bright updates, even if the small-step screen passes earlier. Stop
and report failures for invalid BFGS secant curvature, nonpositive solver,
nonfinite values, a step norm>0.05 or an estimate outside the unit disk.
An exactly identical centre and score can retain the previous solver matrix.
There is no likelihood-increase line search or truth-directed step choice.

## Reusing the first bright iteration

The requested starting centre exactly matches the saved second full pass's
evaluation centre. Its per-row moments and fresh normalization are already
available with unchanged identities. Reuse the142 matching bright rows and
carry all other rows from the original first pass. Do not use the second
full pass's faint moments in the combined score or information. This is the
first of the five requested bright iterations, not an extra initialization.

The resulting first update is
`(0.020762556405984855, 0.0007583995581368543)`.
It is an intermediate result, not evidence of convergence or accuracy.
The remaining four iterations require new bright likelihoods and full-prior
normalization stencils. The cached first iteration is followed by a GPU
replay gate before those jobs start.

## Execution and validation

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_mock20k_bright20_bfgs_20260926_v1`.

Original full-pass root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_mock20k_recenter_originalprior_20260926_v1`.

New code: `scripts/run_v39_bright_bfgs.py`, `scripts/v39_bright_math.py`,
`jobs/job_v39_bright_bfgs.sh`, `tests/test_v39_bright_math.py`. Helpers,
protocol, source hashes and driver are frozen externally. CPU initialization
16721013 completed with20 focused tests passing in15.30s, including the
existing hybrid/BFGS tests and analytic cross-component numerator/cubic
normalizer carry checks. Cached first-update combination passed identity,
coverage and centre checks. Ruff and shell syntax checks pass.

GPU replay16721017 gates four persistent workers16721029[0–3], one V100 each
on th-cl-nv01, via afterok dependencies with invalid-dependency cancellation.
The previous one-GPU ranking probe16720780 completed before these jobs;
combined study use stays within4 GPUs. Fresh normalization should dominate
wall time: approximately1.5–2hours for the remaining four iterations after
startup, based on the preceding20-shard normalization timings, plus queue time.
Replay startup is verified: the original-group evaluation emitted its first
4/560 observations on th-cl-nv01. Scheduled CPU status audit16721087 writes a
snapshot after two hours; it replaces the cancelled old audit16720572 and
does not wake an agent or notify the owner.
Replay16721017 subsequently completed successfully in5m16s: all ten arrays
matched exactly on560 rows. The four production workers are released from
their dependency and awaiting scheduler allocation.
Production startup verified: all four workers16721029[0–3] are running on
th-cl-nv01 and emitted their first shard normalization masses for the second
bright iteration. Those zero-shear masses reproduce the prior preparation.

The source/raw full passes and stopped Newton chain remain unchanged.
Use this root's `chain.json`, `pass_*/result.json`, `preflight.json` and
`failure_*.json` for the continuation status.

## Interpretation limits

The measured-mag20 carry has not been validated against complete reevaluation
at these new centres. Its stable fixed point, if reached, need not equal the
full likelihood's fixed point. Sampling/normalizer convergence is also
unresolved. Report the five-step trajectory and paired draw changes, and
retain the distinction between observed information and solver curvature.
Conditional errors omit carry-approximation uncertainty and uncertainty in
the learned BFGS matrix. No0.3% acceptance is enabled.

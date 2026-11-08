# Training history is not a necessary recipe

## Completed paired-seed audit

The original likelihood-only **factorized** Cartesian flow is a credible
simpler candidate already available, distinct from the older fully joint
Cartesian model that failed m2. Its training uses a single140epoch NLL run,
original eight inputs (plus missing indicators), no added emulator-derived
neighbour score and no response or cut-probability loss. Selected epoch139
minimizes held-out likelihood; see [original training contract](../alignment_factorized_20261105/rule.md#fixed-training-and-evaluation).

On the same full40 scenes, new paired-seed measurements, predicted usability,
generated selection and all nine cuts:

| Recipe | Central m1 (%) | Central m2 (%) | Both-axis nine-cut RMS (%) |
|---|---:|---:|---:|
| Single NLL run, original factorized flow | −0.50484 ±0.13105 | +0.02729 ±0.16848 | 0.39999 ±0.06862 |
| Latest staged recipe | +0.55576 ±0.13143 | +0.21011 ±0.16898 | 0.63457 ±0.11008 |

The simpler model has smaller absolute point bias in16 of18 cut/component
values. Exceptions are m1 at radius>2.8px/magnitude<24.9 and<25.0: absolute
bias worsens by0.07358 and0.15092 percentage points. Thus it does **not** pass
the previously proposed strict0.1pp pointwise margin at every cut. RMS change
is−0.23458±0.14047pp, bootstrap95% interval[−0.46202,+0.06884]; statistical
superiority is not established. These reused development scenes do not give
independent acceptance or training-seed robustness.

The evidence does not justify treating the full four-stage history as
necessary. The final probability-loss continuation improves its immediate
control, but that does not establish net benefit over the initial NLL-only
model. Four-scene intermediate-stage results are separately preserved and
cannot be substituted for the full40 comparison. They are not enough to
identify a uniquely harmful stage.

Recommend using the original NLL-only factorized checkpoint as the simple
training-recipe candidate, retaining latest as comparison. No new distillation
or multi-stage retraining is needed to obtain this candidate. Do not change
production presets or declare uniform performance retention. If further work
is requested, examine the two low-radius m1 tradeoffs and validate this simple
recipe before adding back an auxiliary objective. Introduce inputs at the
start if needed; their late introduction was an experiment, not a requirement.

## Verification and artifacts

CPU17833559 completed0:0 in7s. All112 historical case/axis receipts pass
checkpoint, auxiliary and source/latent checks; original measured sums replay
on the exact cases/cuts. Latest biases and jackknife errors replay against the
completed paired-seed report. Independent scalar point/leave-one-out diagonal
checks pass2016 comparisons. Historical caches without original data hashes
are hashed now; their original bytes are not retrospectively authenticated.
Full tables and uncertainty arrays: [results](results.md#full-40-scene-comparison)
and `report.json`. The audit generates no new model predictions or training.

Cancelled architecture-compression jobs17833512–17833515 and preserved their
partial checkpoints after scope clarification. Broad17788286 remains held;
no automatic follow-up or new training is running for this task.

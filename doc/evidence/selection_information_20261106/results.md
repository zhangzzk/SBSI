# Selection information diagnostic

Disjoint-case diagnostic selection prediction; errors are measured minus predicted.
Test cases: [1000, 1040, 1080, 1120].

| Predictor | Held-out log loss | Squared error | Reference coefficient error |
|---|---:|---:|---:|
| flow | not compared | +0.068330 ± 0.000437 | +0.003599 ± 0.000078 |
| existing8 | +0.237752 ± 0.001500 | +0.068394 ± 0.000407 | +0.002619 ± 0.000128 |
| existing8_capacity | +0.236366 ± 0.001538 | +0.068090 ± 0.000419 | +0.002584 ± 0.000125 |
| plus_scene_score | +0.236119 ± 0.001549 | +0.067954 ± 0.000422 | -0.000010 ± 0.000133 |

- These are selection predictors, not replacement flows or calibrated responses.
- The target conditions on actual usability in all four legs; conclusions do not automatically transfer to per-leg usability or the unconditional population.
- Flow log loss omitted because its64-draw pass probabilities can be exactly0/1; squared error includes Monte Carlo integration variance.
- Bootstrap fixes the fitted nuisance models; four development scenes cannot establish independent acceptance.
- Additional-feature gain is predictive evidence, not a proof of information-theoretic insufficiency.

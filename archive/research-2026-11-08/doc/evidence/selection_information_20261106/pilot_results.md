# Selection information diagnostic

Disjoint-case diagnostic selection prediction; errors are measured minus predicted.
Test cases: [1000].

| Predictor | Held-out log loss | Squared error | Reference coefficient error |
|---|---:|---:|---:|
| flow | not compared | +0.068766 (one test scene) | +0.003446 (one test scene) |
| existing8 | +0.238805 (one test scene) | +0.068863 (one test scene) | +0.002266 (one test scene) |
| existing8_capacity | +0.237453 (one test scene) | +0.068589 (one test scene) | +0.002260 (one test scene) |
| plus_scene_score | +0.237245 (one test scene) | +0.068449 (one test scene) | -0.000353 (one test scene) |

- These are selection predictors, not replacement flows or calibrated responses.
- Flow log loss omitted because its64-draw pass probabilities can be exactly0/1; squared error includes Monte Carlo integration variance.
- Bootstrap fixes the fitted nuisance models; four development scenes cannot establish independent acceptance.
- Additional-feature gain is predictive evidence, not a proof of information-theoretic insufficiency.

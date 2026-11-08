# Photometry learning × neighbour information

## Owner-authorized four-way comparison

Owner 2026-11-07 asks to work on improved conditional learning and richer neighbour inputs, following the completed capacity/information diagnostic. Start all four arms from the original nine-cut regularized checkpoint `888fb04402546518657386a097a33ba37639abc2745c99724f96643a2cf21e2e`. This differs from the completed conditional-cut experiment, which started from a previously neighbour-trained checkpoint; do not mix those baselines.

| Arm | Additional neighbour channel can learn | Extra conditional-cut probability loss |
|---|---|---|
| control | No | No |
| weighted | No | Yes |
| neighbour | Yes | No |
| both | Yes | Yes |

Augment the original architecture with a zero-weight extra input exactly as in the earlier neighbour model. Every arm initially has exactly the same physical draws and density as the original model. All arms receive identically standardized contexts for implementation consistency. For input-off arms, freeze every new photometry input weight at exactly zero, making predictions functionally invariant to the extra value. For input-on arms those photometry weights can learn. Keep ALL shape-factor weights, including its new zero-weight input connections, fixed in every arm. Verify density and draw replay before fitting, score invariance for input-off models after fitting, and identical conditional shape density at fixed photometry, including when the score is perturbed.

## Input, data and objective

The extra input is the existing frozen-emulator neighbour coefficient sum at reference radius 3.5 pixels and magnitude 24.5, derived entirely from truth scene geometry/properties. Cached features and training-only standardization remain fixed. The input carries no measured photometry label. This tests a useful detailed scene summary; it is not a new measurement, a unique physical-feature identification, or a neighbour-shear-changing input channel.

Use the original 160 training scenes and all their usable rows. Preserve the original NLL and full primary response penalty, numerical batches, seeds and nine target cuts: radius >2.8/3.0/3.2 pixels crossed with magnitude <24.9/25.0/25.1. Train photometry only for exactly 64 AdamW updates at 1e-5 with no decay. NLL batch 8192; primary-response groups 32768 pairs with four antithetic draws; original response weight 0.001; photometry-gradient norm clipping at 5. The four arms share row/latent streams and optimizer settings. Only the allowed extra weights and the probability-loss presence differ.

The added objective is the per-row/per-cut mean `(q_A-y)*(q_B-y)`, with two independent four-draw antithetic banks on the same 32768 rows and soft-cut widths 0.05 pixels/magnitudes. It targets conditional probability, without the integration-variance bias of a squared single-bank prediction. Keep the existing verified estimator and row/latent seeds. Do not average rows before multiplying residuals or clamp the finite score.

Recompute one training-only auxiliary weight at the untouched original start, with input-off photometry parameters: four batches, weight = 0.25 times RMS original-objective gradient norm divided by RMS auxiliary gradient norm. Use that SAME scalar in weighted/both; do not recalibrate separately with the extra score or choose it from validation/coherent outcomes. Every arm retains its own fixed step-64 endpoint. Resume states preserve optimizer, parameters, history and source provenance.

## Evaluation and reporting fixed before training

Own-data checks use the same 500k rows per split, original 40 validation scenes, original case/leg mapping and 64 portable draws. Report all nine hard-cut probability scores (independent 32-draw banks), original soft score, 0.025/0.1-width sensitivities, photometry and conditional-shape NLL, passing mass, selected shape numerators/means, and selected-neighbour score. These checks do not select the endpoint. Shape parameters are fixed, but selected-shape behavior can change through photometry and selection.

Run the exact four coherent development scenes 1000/1040/1080/1120 on both axes with the unchanged C8 classifier and physical emulator, paired 64 draws and original 1024-parent blocks. Keep independent per-leg selection/normalization and all three usability/selection modes. First case/axis passes simulation replay, measured-count identity, reference-score replay and separate-versus-pooled emulator checks before the remaining seven tasks start. Model components are prediction contributions, not independently identified physical component errors.

Report every arm, all nine target cuts, both axes, per-scene and pooled m, nine-cut RMS, full flow/emulator contributions, paired scene bootstrap/jackknife errors, and the interaction `(both-neighbour)-(weighted-control)`. Show signed and absolute bias changes; accept internal cancellation but do not select cuts or checkpoints from the results. Four reused scenes do not establish independent acceptance or robust 0.3% calibration. This bounded continuation is not a test of training to convergence.

## Execution and correctness gates

One CPU fixture job checks the independent-bank score/gradient, unequal-case reducers, all four factorial contrasts and input-freezing rules. One GPU allocation then checks exact original-model augmentation, real-model gradients, training-only loss calibration, frozen shape behavior and exact reload; trains all four arms; runs own-data evaluation; and completes the first coherent field/axis. Three dependent persistent workers cover the remaining seven tasks, followed by a CPU report/export. Use `inter` by default, with `cip` fallback, at most three GPUs concurrently (within the six-GPU study ceiling), four CPUs per GPU allocation, 32 GB host memory. Pack tasks; preserve batch sizes and per-case seeds. Large outputs remain on the project filesystem.

The broad array 17788286 remains held. No new images, inferred offsets, production replacement, or restart of the stopped automated-follow-up loop. Source hashes gate all stages; numerical failures stop dependent stages. Scientific outcomes do not change the declared evaluation set. Monitor startup and the first meaningful progress; once a long job progresses normally follow the repository's long-job handoff policy.

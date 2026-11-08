# Nine-cut response plus selected-neighbour moment

## Frozen experiment

Owner asks to continue improving the flow after the selection--neighbour
association error was found on its own training and validation data. Start two
arms from the same frozen neighbour-conditioned NLL checkpoint. Both receive
the existing64 updates of NLL plus weak nine-cut full-matrix response penalty.
The weighted arm adds the training-data moment penalty below; the matched
control gives it zero weight. Keep the original classifier and disk emulator.
No constant-shear measurements enter training or choose endpoints/weights.

Use the original radii2.8/3/3.2 pixels crossed with magnitudes24.9/25/25.1,
sigmoid softness0.05 pixels and0.05 magnitudes. All settings, streams and
checkpoint hashes are frozen in manifest.json before training. Use the fixed
step64 endpoints, AdamW1e-5 with zero weight decay, norm5 gradient clipping,
NLL batch8192 and independent32768-row replicas with four antithetic draws.
The original response penalty weight remains0.001; the additional weight is
0.001. There is no hyperparameter search or response-selected checkpoint.

## Additional objective

B_i is the existing frozen truth-scene neighbour coefficient at reference
photometry3.5px/24.5mag. From all original training NLL rows (both g0/g05 usable
legs), compute per-cut measured soft mass D_c and selected coefficient b_c.
For generated and measured soft selections w and w_obs, use contributions

    a_ic = (E_draw[w_ic] - w_obs,ic) / D_c
    q_ic = (E_draw[w_ic] - w_obs,ic) * (B_i - b_c) / D_c.

Their full-population means vanish exactly when selection mass and selected
mean B match the measured targets. Use scales0.003 for the fractional mass
residual and0.001 for the selected-coefficient residual. The added objective
is half the mean product of scaled mean residuals from two independent row
and latent replicas. Replay its gradient; test it against direct autograd.
Subtracting the measured row contribution is a loss estimator, not an offset
to predictions. Its derivative is zero; no post-processing correction exists.

The target includes the original usable-row population, not a truth-property
cut or a common-leg intersection. The shear-response target continues to use
its original paired population. Diagnose hard cuts separately; matching this
reference-coefficient moment is not proof of full joint shear calibration.

## Validation and execution

Compute training targets once with a streaming CPU pass inside the first
allocation. Validate input-score alignment and target reconstruction. Check
photo-factor/joint-draw equality and direct/replayed moment gradients, retaining
both independent replicas. Run both arms sequentially on the same GPU, with
identical NLL/response streams and per-eight-step resumable optimizer states.
Record a matched initial-response probe and each final probe; compare the
control with the previous identical-protocol endpoint as a numerical audit.

The first persistent GPU job performs preparation, both trainings and the
first case/axis evaluation of both endpoints. It gates three persistent GPU
workers, each processing multiple remaining case/axis tasks with both models
and emulator kept loaded. Retain individual case outputs and verify receipts
before skipping completed work. Original1024-parent blocks,64 portable CRN
draws, all14 saved diagnostic cuts, per-leg flags/selection and response
definitions remain unchanged. Evaluate all four declared scenes on both axes,
with paired case bootstrap/jackknife errors and the selected-neighbour score.
The original broad evaluation remains held. Maximum concurrency is three GPUs
for this trial, within the authorized six-GPU study ceiling.

Pool emulator tree-cell lookup across the two endpoints and shear legs within
each unchanged1024-parent block. The emulator has the same fixed truth pairs
and only photometry-dependent output inputs. Before using this optimization,
require agreement to separate evaluations on the first complete block
of every case/axis, at2e-14 absolute/relative tolerance; record the maximum
difference in each receipt. The implementation sums neighbour predictions in
200k-row batches, so pooling can change floating-point addition grouping.
This tolerance is declared before the pilot and permits only roundoff. Flow draws, neighbour
lists, transport and accumulated moments are unchanged.

Evaluate own-data hard-cut score/pass errors on the same500k training and500k
validation rows and the same latent banks as the preceding diagnostic. Use
these and validation NLL as diagnostics, not endpoint selection criteria.
Pilot expansion is gated only by correctness and provenance checks, not bias.
Long normally progressing stages use a bounded local status check rather than
continuous polling. No production model or likelihood configuration changes.

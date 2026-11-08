# Why the flow's shape response moved

## Measured effect before the new training

The neighbour loss directly uses only generated size and flux. Its direct
conditional-shape gradient was exactly zero in the original replay/autograd
gate. The earlier optimizer still trained both factors with likelihood and the
nine-cut response loss and clipped the **combined** gradient; the saved
pre-clipping gradient norms exceeded the threshold 5 in all 64 updates of both
arms. The response gradient depends on generated photometry, and a change in
photometry gradients can also change a shared clipping factor. These are
available routes for the shape weights to move differently. The swap is an
endpoint comparison; it does not isolate which optimizer route dominated.

At the central cut on four fixed scenes, actual per-leg usability and generated
selection, the old factor swaps give these **changes in predicted flow response**
when adding the neighbour moment (delete-one-scene jackknife SE):

| Intervention | R11 flow change | R22 flow change |
|---|---:|---:|
| Change photometry only, keep old nine-cut conditional shapes | −0.002268 ± 0.000083 | −0.002250 ± 0.000085 |
| Change conditional shape only, keep old photometry | −0.000983 ± 0.000016 | −0.002980 ± 0.000013 |
| Change both | −0.003246 ± 0.000090 | −0.005219 ± 0.000077 |

The same conditional shape law may produce a different selected average shape
when the photometry fed into it and the galaxies passing the cut change
differently on the plus and minus legs. Repeating the first row with identical
*measured* membership for both models reduces the flow change to −0.000188 /
−0.000146. This indicates sensitivity to generated membership and its
correlation with generated shape; it is not an additive causal selection budget.
The accompanying emulator increase in the first row is +0.002483 / +0.002489;
the net **simulation-minus-model residual change** is −0.000215 ± 0.000088 /
−0.000239 ± 0.000090. This is a small improvement at the central cut and
does not establish better calibration across cuts: the sign of the photo-only residual change reverses for three of nine cuts on each axis. At the center the unchanged nine-cut m1/m2 values are +1.056%/+1.379%; the frozen-shape endpoint swap gives +1.032%/+1.352%, versus +1.130%/+1.701% with both updated factors. See the
[saved contrasts](../neighbour_factor_swap_20261107/cut_contrasts.tsv) and
[original optimizer](../neighbour_moment_guard_20261106/train.py).

## New experiment: train with the shape model locked

The [protocol](rule.md#paired-optimizer-intervention) fixes the original
nine-cut pre-moment conditional shape model for the entire continuation.
Control and weighted start from the same checkpoint and share all samples,
latent draws and optimizer settings. The new loss weight is their only
difference. The [jobs](jobs.json) cover training, own-data checks, a first
constant-shear replay and the gated seven remaining scene/axis replays. No
production changes, extra simulations, or automated thread follow-ups.

## Completed frozen-shape comparison on both axes

All eight field/axis pairs finished; Slurm 17815619, 17815620 (all three workers) and 17815621 exited 0:0. The separate per-scene reduction [replays every pooled contrast and its jackknife SE](../neighbour_crossing_response_20261107/frozen_scenes.json). The zero-shear selected-neighbour score improves on validation by 0.002605 ± 0.000027 relative to the matched frozen control. It **does not improve joint calibration around the target** on these four development scenes.

Weighted minus matched frozen control, simulation minus model, central radius >3 px/MAG_AUTO <25. Below, all ± values are **four-scene delete-one jackknife SEs**, not the bootstrap SE reported in the legacy-format [summary table](results.md). Signed absolute-unit response changes are distinct from multiplicative bias m.

| Usability / cut | Change in model flow R11 / R22 | Change in model emulator R11 / R22 | Change in residual R11 / R22 | Nearby cuts with smaller absolute pooled residual, by axis |
|---|---:|---:|---:|---:|
| Predicted / generated | −0.003256 / −0.003248 | +0.002997 / +0.003000 | +0.000259 ± 0.000049 / +0.000248 ± 0.000041 | 3/9, 3/9 |
| Actual / generated | −0.003271 / −0.003299 | +0.002897 / +0.002908 | +0.000374 ± 0.000039 / +0.000391 ± 0.000065 | 1/9, 1/9 |
| Actual / measured | −0.000824 / −0.000811 | +0.000249 / +0.000264 | +0.000575 ± 0.000010 / +0.000547 ± 0.000008 | 0/9, 0/9 |

At the central cut, the signed residual rises in all four individual scenes on each axis in all three modes. The per-scene statement refers to signed residuals; the cut counts refer to pooled absolute residuals. With predicted usability, central m1 moves from +0.741% to +0.770% (weighted endpoint absolute jackknife SE 0.720 percentage points), m2 from +1.117% to +1.145% (0.483 points). With actual usability and generated photometry, m1 moves from +0.973% to +1.015% and m2 from +1.286% to +1.331%. The nine-cut RMS m rises with predicted usability by +0.0611 ± 0.0062 / +0.0385 ± 0.0098 percentage points; actual usability with generated selection gives +0.0704 ± 0.0060 / +0.0496 ± 0.0105 points. The nearby cuts and the two axes share scenes. Measured selection still supplies generated photometry to the shape and emulator models. These endpoint changes are conditional on the tested models and fields, not standalone component errors, an independent validation or a model-selection rule. No retraining or larger evaluation is indicated by the static score improvement alone.

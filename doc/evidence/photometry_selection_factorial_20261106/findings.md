# Selection, photometry and neighbour response

## What the factorial tests

The target cut is radius>3 pixels (0.6 arcsec), magnitude<25. All branches use
the weak-nine flow and the same original emulator, actual common usability,
four opposite-group edges and equal within-group normalization. They measure
the shape-response component. Measured neighbour-induced selection remains a
separate term. Definitions follow [response estimators](../../CONVENTIONS.md#6-response-estimators).

For one edge, the model quantity has the form

    R = sum[weight * predicted shape change] / (0.04 * sum[weight]).

Changing only selection flags therefore does not test whether the selected
distribution of measured radius/flux is correct. The numerator depends jointly
on the mask, photometry, conditional shape and each object's neighbour scene.
The emulator's standalone mean residual on measured inputs does not constrain
this different joint moment evaluated on generated inputs.

The G/G branch retains each generated selection mask on its own draw. M/M
conditions flow shapes on the measured photometry and applies measured weights.
The crossed G/M and M/G branches are interventions that deliberately break some
associations. They are not usable corrected models. Report both substitution
orders and their interaction; neither order defines a unique physical budget.

## Controls and uncertainty

The full-parent block order, original64 portable antithetic draws and original
GPU family are preserved. All generated masses and response numerators must
replay the old four-state calculation. The reducer verifies that replay in every
bootstrap sample, all factorial identities, intrinsic-bin additivity and zero
membership difference for the uncut branch. Original four-scene bootstrap
resamples remain unchanged. Individual-scene results, delete-one-scene jackknife
SEs and half-bank differences supplement them. These are fixed-model development
checks; neither uncertainty estimate includes training or model uncertainty.

The scalar emulator coefficient is also averaged with exactly the same weights.
Its contrast plus the remaining finite shape-transport contrast must reproduce
each response contrast. This locates input sensitivity without adjusting the
emulator or applying empirical correction factors.

## Evidence and subsequent localization

All four GPU workers and full reducer17806280 completed exit0 and passed every
replay and identity gate. The [four-scene report](results.md) gives absolute
shape-response errors, simulation minus model:

| Photometry inputs | Selection weights | Shape-response error +/- case-bootstrap SE |
|---|---|---:|
| Generated | Generated | +0.005217 +/-0.001492 |
| Generated | Measured | +0.005391 +/-0.001546 |
| Measured | Generated | +0.004562 +/-0.001500 |
| Measured | Measured | -0.001339 +/-0.001549 |

Photometry alone raises prediction0.000655 +/-0.000055; measured weights alone
change it-0.000174 +/-0.000126. Replacing both raises it0.006556 +/-0.000171,
with interaction0.006075 +/-0.000032. Both substitution orders are preserved in
the report. The diagonal change is positive in all scenes (0.006089–0.006898);
its delete-one-scene jackknife SE is0.000199, versus bootstrap0.000171. The
interaction jackknife SE is0.000037. These are uncertainties on paired model
contrasts, not absolute calibration errors.

Scalar emulator coefficient change0.006547 +/-0.000172 accounts for almost the
entire diagonal contrast; remaining finite shape-transport geometry changes
0.000009 +/-0.000011. Thus the main effect is input-dependent response weighting,
not a new conditional-shape alignment failure. Antithetic half-bank difference
in the contrast is-0.000005. The M/M result changes0.000002 +/-0.000011 from the
previous independent draw bank. Neither diagnostic suggests material draw noise
at the scale of this contrast.

Measured neighbour-induced selection is a separate+0.002925 +/-0.000418;
including it leaves M/M residual+0.001586 +/-0.001842. This is the common-U,
within-group diagnostic, not the globally normalized coherent error budget.

The follow-up
[photometry-bin protocol](../photometry_response_bins_20261106/rule.md#exact-identities-and-controls)
tests whether the diagonal difference reflects changed radius/magnitude-bin
fractions or different mean responses within those bins. Multiple resolutions
and optional intrinsic-size strata expose binning dependence. Even a persistent
within-bin term cannot by itself distinguish density underfit from insufficient
conditioning information. The [completed four-scene replication](../photometry_response_bins_20261106/findings.md#four-scene-replication-complete)
has a dominant within-bin term in every scene. Broad17788286 remains held.

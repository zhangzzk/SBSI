# Response mismatch after matching photometry bins

## Four-scene replication complete

All remaining workers17806587/17806588/17806589 completed exit0; reducer17806590
completed exit0 in3s. Every original factorial bootstrap response and every
distribution/conditional-mean identity replays. The
[full report](results.md) gives, for fine radius/magnitude bins:

| Model-response change | Value +/- case-bootstrap SE |
|---|---:|
| Different bin fractions | -0.000204 +/-0.000281 |
| Different response within bins | +0.005938 +/-0.000241 |
| Exclusive support | +0.000822 +/-0.000011 |
| Total measured-minus-generated input contrast | +0.006556 +/-0.000171 |

The within-bin term is positive in every scene:0.005588,0.006441,0.006418,
0.005370 for1000/1040/1080/1120. Adding intrinsic-size strata gives pooled
0.006001 +/-0.000277. At coarser resolution the term remains dominant,
0.006337 +/-0.000225. Exclusive support is preserved; fine common-support mass
is about99.995–99.998% generated and99.704–99.717% measured.

This verifies the pilot pattern across all four scenes. Residual photometry
variation inside finite bins still prevents attributing it uniquely to scene
mix. The [fixed-cell follow-up](../photometry_fixed_cell_20261106/rule.md#estimator-and-intervention)
holds the emulator's photometry exactly identical within each fine cell while
preserving selected membership and original truth pairs. It is a controlled
model diagnostic, not a change to rendering or trained models.

## Pilot: the within-bin term dominates

Case1000 GPU17806566 completed exit0 and reducer17806572 passed every raw-moment
and bootstrap replay gate. This is a single-scene diagnostic, with no estimated
population SE. The [report](pilot_results.md) decomposes the exact0.006089
generated-to-measured input change using the protocol's fixed bin resolutions.

For fine radius/magnitude bins, the distribution term is-0.000286, the within-bin
response term+0.005588, and exclusive support+0.000787. After coarsening, the
terms are+0.000141,+0.005904,+0.000044. Adding intrinsic-size strata leaves the
within-bin term dominant:0.005658 at fine resolution and0.005987 at coarse.
All terms sum exactly; no unsupported-bin mean is invented or extrapolated.

Thus different fractions across these photometry bins do not explain the
pilot deficit. At similar radius and magnitude, generated draws and measured
objects carry different average neighbour responses. This strengthens the case
for a mismatch in their association with intrinsic/neighbour properties, but
does not yet establish that particular missing flow inputs caused it. Residual
within-bin photometry differences and conditional-density underfit remain
possible. The original emulator weights have not changed.

Fine bins expose more exclusive support, partly because measured endpoint
weights are averaged across shear states. A target selected at one endpoint can
contribute through its other endpoint outside the cut. Coarsening absorbs some
of that boundary support. Retain the support term and multiple resolutions;
do not select the smallest exclusive-support result as an acceptance criterion.

## Replication provenance

The passed pilot gated the remaining three original cases. Each worker retained
its parent hardware family, draw bank and per-size replay checks. Full report
17806590 also waited for the parent four-scene factorial. This remains a
development diagnostic; held broad evaluations stay paused.

# Grid-based response regularization: recent and historical results

## Scope and terminology

Located on 2026-11-05 from preserved logs and result artifacts; no runs restarted.
Property-grid regularization averages response residuals within cells of galaxy
properties. The later 16-cut/9-cut guards instead constrain a list of measured
selection thresholds; they are different experiments. The results below keep
training-family response proxies separate from coherent constant-shear tests.
All percentages use the stated simulation/model response-ratio convention
unless an older reciprocal convention is explicitly identified. None of these
comparisons alone establishes robust end-to-end calibration.

## Successful constant-shear results omitted from the initial index

Correction after the owner's follow-up: the initial index overrepresented
failed follow-ups and omitted successful older grid-regularized models. It
must not be read as evidence that grid regularization never worked.

The August 24 original mixed-shear **E** flow used a 6x6x5 true-magnitude,
true-size and r_blend grid with a full 2x2 shape-response penalty, lambda 450.
It trained NLL on both g=0 and g=0.05 legs and used an SWA average of epochs
73–80. Its photometric coupling penalty was off. The response target used
160 training cases; the full-matrix penalty was a per-cell Frobenius error,
not the later covariance-normalized grid objective. Lambda values across
these recipes are therefore not directly comparable.

On the fixed 100-case ConstGold population, seeds 501–504 gave m of
+0.39017%, +0.05749%, +0.35380% and +0.13665%, respectively (case SE about
0.126 percentage points each). Averaging flow responses gave
**m=+0.23433% +/-0.15040 percentage points**, combining case and seed-mean
terms in quadrature. The frozen 16-seed V3 comparison was
**+0.34883% +/-0.17727 percentage points** on the same population.
These tests restricted truth magnitude to 18–25.8 and truth size to
0.5–1.5 arcsec; they are not the later unrestricted-parent, independently
measured-cut test. A cross-response residual remained for E.

Sources: [E recipe](../../archive/research-2026-09-14/doc/WORKLOG_FULL.md#2026-08-24-cont214--200-case-mixed-shear-cde-flow-experiment-launched),
[four-seed result](../../archive/research-2026-09-14/doc/WORKLOG_FULL.md#2026-08-24-cont223--four-seed-original-e-closes-constgold-at-0234---0150),
[population and frozen V3 comparison](../../archive/research-2026-09-14/doc/WORKLOG_FULL.md#2026-08-24-cont216--arm-e-transfers-to-constgold-at-0390-m-with-residual-cross-response).

The newer retained models also had good constant-shear development results:
V3.6 main80 m=-0.199979% +/-0.263649 percentage points, and V3.9 main80
m=+0.020770% +/-0.265152 percentage points. Both apply independent per-leg
measured radius>0.6 arcsec and magnitude<25.8 cuts. V3.9 has no truth-property
cuts. Their pipelines include common-descent probability/response refinement;
the V3.9 stage-zero qualification below is essential. Their
quoted errors are case-bootstrap only; no independent acceptance was completed.
Sources: [V3.6 evidence](../JOINT_CALIBRATION.md#retained-evidence),
[V3.9 evidence](../V39_MODEL.md#development-evidence),
[F8 training stages](../../archive/research-2026-09-25/doc/V36_FEATURE_SIMPLIFICATION.md#f8-execution-record).

## V3.9 training provenance: measured-cut grid and retained step zero

The retained F8 flow uses a **measured-threshold grid**: radius thresholds
2.8/3.0/3.2 pixels and magnitude thresholds 25.6/25.8/26.0, giving nine joint
cuts plus three radius-only, three magnitude-only and one global guard. Softness
is 0.05 pixels and 0.05 mag. Initial training fits NLL, then adds four guard
updates per epoch with weight 1 and response scales 0.01 globally/0.02 for cuts.
These are selected-shape response constraints, not a true-property-cell grid.

The saved F8 target bank has `n_strata=1` and 16 targets. Its harmonized
control-variate refinement selected step 2, with validation NLL changing from
-4.7450567124 to -4.7441691639 (within the 0.001 nat/row budget). Crucially,
the subsequent common-descent probability/response stage selected **step 0**:
its baseline and final guard loss are both 0.4172889123. Thus the retained
flow passed through that stage but did not adopt a common-descent update.
The final receipt SHA is the pinned V3.9 flow `bd51dbbbeeb7...`.

Verified directly against `experiment.json`, `base/flow/protocol.json`,
`harmonized/refinement/completed.json`, `joint/per_leg_guard_train_targets.json`
and `joint/joint_probability_refinement_full/completed.json` under
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/`.
The frozen trainer, per-leg target builder and common-descent runner hashes
match the experiment manifest. No weights or model identities were changed.
The successful +0.020770% +/-0.265152 pp result is the joint F8+C8+disk result,
not an isolated measurement of the benefit from any one training stage.

## November 4: GalSBI property grid with the measured cut inside the response term

Recipe: 216 cells, 6 bins each in true magnitude, circularized true size and
`nbr_close_100`. Edges are training-pair quantiles weighted by baseline soft
pass probability. Each cell constrains the soft selected response for measured
FLUX_RADIUS>3 px and MAG_AUTO<25, with case-jackknife SE weighting per component.
Lambda 1, 12 cells x 128 pairs x 16 draws in every NLL step. Training uses 23.3 M
pairs; validation 5.8 M pairs/40 cases. Selected grid epoch 11 of 36.

These are **flow-alone soft-cut m_eq**, not hard-cut coherent constant-shear m:

| Cut, magnitude<25 | Density only | Original 16-cut guard | Property-grid-with-cut |
|---|---:|---:|---:|
| Radius>2.8 px | -2.86 +/-0.35% | -1.63 +/-0.36% | -8.32 +/-0.33% |
| Radius>3.0 px | -0.12 +/-0.34% | +0.80 +/-0.35% | +0.79 +/-0.35% |
| Radius>3.2 px | +0.58 +/-0.32% | +1.08 +/-0.32% | +9.31 +/-0.35% |

The declared 21-cut target RMS is 6.58%, maximum absolute error 9.81%; held-out
RMS 11.39%, NLL cost +0.229 nat/row. It fails all four declared gates. This 21-cut
target summary differs from `rms_m_eq_guarded` in the JSON, which summarizes
the original 16-cut list. The original 16-cut RMS in that artifact is 10.13%.
Training gradients were 50–200 x the NLL gradients and every update clipped.
The sharp threshold dependence is consistent with exploiting selection near
the cut; the original log explicitly notes that a radius histogram did not
independently establish that mechanism. No constant-shear result for this arm.

Sources: [November 4 work log](../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-04--grid-response-term-with-the-measured-cut-inside-it-fails-the-gate-badly-size-cut-edge-exploited),
[saved check_grid.json](../../.claude/worktrees/sampler-recipe/doc/evidence/galsbi_guard_strength_20261103/check_grid.json).
Checkpoint: `/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_guard_strength_20261103/grid_cut/guard/selected.pt`,
SHA256 `62da726b4369820ad327d5bf6844733de79e0bdbd147993824da0b235c61a822`.

## September 16: fixed-g0 truth-property versus measured-anchor grids

Both are 6 x 6 x 6 continuations of fixed-g0 NLL epoch 140, lambda 1, supervising four
shape and two physical-radius response components with full within-cell
case-jackknife covariance. Truth grid: true magnitude x circularized true size
x cached baseline r_blend. Measured grid substitutes baseline MAG_AUTO and
FLUX_RADIUS for the first two axes. The cohort is fixed by g0 magnitude<25.8
and radius>3 px; its IDs carry to the sheared leg without a new selection.

100-case coherent ConstGold response results:

| Flow | Actual per-leg usability m | Classifier-weighted m |
|---|---:|---:|
| Staged grid-free reference | -1.2920 +/-0.2168% | -1.7333 +/-0.2152% |
| Truth-property grid | -1.5095 +/-0.2167% | -2.9591 +/-0.2128% |
| Measured-anchor grid | -1.2979 +/-0.2173% | -2.5969 +/-0.2136% |

Neither improved calibration. The measured-grid checkpoint selection was
integration-noise dominated: score -14.25 with MC SE 98.71. This fixed-g0
conditioning differs from November's independent per-leg measured selection.
Source: [September 16 result and recipe](../../archive/research-2026-09-20/doc/WORKLOG.md#2026-09-16--constgold-property-panels-remade-for-both-6x6x6-grid-flows).

## September 11–13: circularized shape grids and shape-plus-radius grids

The older grid objective supervised **uncut matched-usable pair responses**,
without a measured selection inside the penalty. Truth support still included
18<true magnitude<25.8 and 0.37<semi-major Re<1.5 arcsec. Grid axes were magnitude,
size and cached r_blend; circularized size replaced semi-major size. Shape-only
used 180 cells; the subsequent shape+radius trial used 216. The common comparison
below is an equal-cell average on the original 180-cell definition, not a pooled
selected catalogue response. Cases 160/40; uncertainties are case-jackknife SEs.

| Flow | Shape trace response-ratio proxy | Full validation NLL |
|---|---:|---:|
| NLL-only epoch 320 | +1.6874 +/-0.3342% | -2.485437 |
| Shape-only grid lambda 0.1, epoch 169 | +1.4952 +/-0.3332% | -2.467832 |
| Shape+radius covariance grid lambda 1, epoch 146 | +1.8019 +/-0.3344% | -2.425763 |
| Shape+radius identity-weighted grid lambda 1, epoch 182 | +1.581 +/-0.334% | -2.481027 |

Shape-only improved the covariance-weighted shape score 40.5%, but moved the
mean response proxy only 0.192 percentage points and worsened radius response.
The 216-cell covariance trial fitted mean radius response well but retained a
shape deficit and cost 0.05967 nat/row versus NLL-only. Removing covariance
recovered much of the likelihood at the cost of a worse cell-response fit.
Equal numerical lambdas are not equal effective regularization strengths.

The apparently successful **covariance-grid training-scene** result was
actual-U self-response m=-0.025 +/-0.139% with magnitude<25.8 and measured
radius>=0.75 arcsec, without the measured ellipticity cut. It did not transfer:
the subsequent 100-case coherent test gave actual-U m=+1.153 +/-0.127% and
classifier-weighted m=+1.017 +/-0.128% under those cuts. The grid flow was rejected.
With the additional measured |e|<0.6 cut, coherent m was +1.613/+1.854%,
respectively. These cuts and older FS2 catalogue are distinct from November's.

Sources: [shape-only final comparison](../../archive/research-2026-09-14/doc/FLOW_CIRCULARIZED_CONDITION.md#completed-follow-up-comparison-2026-09-11),
[covariance-grid audit](../../archive/research-2026-09-14/doc/FLOW_CIRCULARIZED_GRID_SHAPE_RADIUS.md#covariance-branch-final-review--2026-09-12),
[identity-weighted audit](../../archive/research-2026-09-14/doc/FLOW_CIRCULARIZED_GRID_SHAPE_RADIUS.md#no-covariance-final-review--2026-09-12),
[coherent rejection](../../archive/research-2026-09-14/doc/FLOW_TRAINING_SELF_SELECTION.md#coherent-covariance-flow-rejection--2026-09-13).

## August 10: refining magnitude/blend cells while removing the size axis

Historical 10 x 1 x 16 and 12 x 1 x 16 magnitude/size/r_blend grids improved magnitude
and blend profiles but failed the size-profile check. Four matched seeds:
ConstGold m=+0.7414 +/-0.3256% and +0.8213 +/-0.1659%, versus reference
+0.7760 +/-0.2059% (uncertainties here are **seed SEM**, not case bootstrap).
Neither changed global m significantly. Half-shear size-profile RMS rose from
0.522% to 4.252%/4.484%; ConstGold size RMS rose from 0.787% to 4.855%/4.917%.
The attractive seed 501 result m=-0.146% did not persist across seeds.
Source: [four-seed no-size-grid comparison](../../archive/research-2026-09-14/doc/WORKLOG_FULL.md#2026-08-10o--four-matched-seeds-reject-the-no-size-designs).

## Related measured-cut-grid guards, distinct from property grids

November 4's 9-AND-cut guards used radius thresholds 2.8/3.0/3.2 px crossed with
magnitude 24.8/25.0/25.2. At radius>3/MAG_AUTO<25, 4 and 40 guard steps gave
m_eq +0.59 +/-0.35% and +1.36 +/-0.35%; both failed their target/held-out gates.
The band guard was better across nearby cuts but still failed its declared
target gate. These should not be conflated with the 216-property-cell arm above.
Source: [AND-cut guard comparison](../../.claude/worktrees/sampler-recipe/doc/WORKLOG.md#2026-11-04--and-only-guard-9-radius-and-magnitude-cuts-at-mag_auto-248250252-fails-the-gate).

# Frozen-factor localization

## Final conclusion: both shear axes

The conditional-shape factor carries most of the change from the NLL endpoint to
the nine-cut response-regularized endpoint. At the target cut it accounts for
96% of the joint residual shift in R11 and 94% in R22. Across all nine cuts, both
axes, and all three usability/selection modes, the pooled share is 91–97%.
The shape-only changes in the flow contribution itself (−0.011389, −0.008427)
closely track the corresponding own-data training probes (−0.011264, −0.008445).
The shape response change transfers to these scenes; a change in which galaxies
pass the cut cannot account for most of it.

The **added neighbour constraint** has a different effect. Updating only the
photometry raises the model emulator contribution by +0.002483/+0.002489
(R11/R22) but lowers the flow contribution by −0.002268/−0.002250. These
nearly cancel on each axis. At the central cut with actual usability, the
complete residual shift is larger in R22 than R11 by **+0.002188 ± 0.000155**.
The shape-only axis difference is **+0.002215 ± 0.000021**; the photometry-only
difference is −0.000024 ± 0.000160 and the interaction −0.000003 ± 0.000003.
This uses the same scenes on the two axes and the same deletion on both sides
of each jackknife contrast.

| Change, actual usability and generated selection | Model flow change R11 / R22 | Model emulator change R11 / R22 | Joint residual shift R11 | Joint residual shift R22 |
|---|---:|---:|---:|---:|
| NLL → response penalty | −0.011165 / −0.008188 | −0.000885 / −0.000648 | +0.012049 ± 0.000074 | +0.008836 ± 0.000015 |
| Response penalty → added neighbour constraint | −0.003246 / −0.005219 | +0.002594 / +0.002379 | +0.000652 ± 0.000097 | +0.002840 ± 0.000084 |

At fixed photometry and selection, a shape-factor swap leaves the neighbour
coefficient unchanged, but changes the base ellipticity to which the disk
transport is applied. Of the +0.002215 shape-only axis difference, +0.001997
comes from the flow and +0.000217 from this transport dependence (residual
sign). The gains of the two diagonal components depend oppositely on
e1²−e2² at zero velocity; see [the implemented transport](../../../sbsi/disk_response_transport.py).
The table uses the original finite shears, not that zero-velocity approximation.

**Scene, cut, and flag checks.** The added constraint gives a larger shape-only
residual shift in R22 than R11 in every one of the four scenes at every nearby
cut in all three modes; the individual differences range +0.002136 to +0.002435.
At the target cut, the shape-only R22 shifts are +0.003090 with predicted
usability, +0.003089 with actual usability, and +0.003111 with actual usability
and measured selection. All three modes use generated photometry for the model
predictions. In contrast, the photometry-only residual change switches sign
across cuts on both axes. The swaps do not support freezing shapes as a
sufficient fix.

Galaxies with **intrinsic** circularized radii 0.15–0.40 arcsec contribute
+0.001833 ± 0.000021, or 82.8% of the shape-only axis difference. They already
make up 75.7% of the sample passing the **output** cut (FLUX_RADIUS > 3 pixels,
MAG_AUTO < 25). This localizes a contribution to the selected mean; it does
not establish a large per-object error or make size itself the cause.

The measured-minus-generated **selected reference-neighbour coefficient** on
the common-four-usable cohort falls from +0.003112 to +0.000866 (paired change
−0.002246 ± 0.000017). It is identical for products with the same photometry
even when their shear responses differ. This is direct evidence that agreement
in that proxy does not ensure agreement in the coherent shear response.

Absolute calibration uncertainties are larger than uncertainties on paired
model changes. At the target cut (actual usability, generated selection), m1/m2
are −0.284% ± 0.694 / +0.390% ± 0.474 initially, +1.056% ± 0.695 / +1.379% ± 0.478
with the response penalty, and +1.130% ± 0.704 / +1.701% ± 0.482 with the added
constraint. Errors are percentage points. All SEs in this section use four
reused development scenes, fixed endpoints and common draws. The swaps identify
which fitted factors changed, not isolated physical component errors or which
optimizer gradient caused the change.

All eight case/axis evaluations and the Slurm report finished successfully. All
56 result files passed checksum/replay checks; unchanged-source maximum response
discrepancy was 2.93e-12. See the [scene and size tables](scene_size_breakdown.md#central-residual-and-component-changes),
[individual cut results](cut_contrasts.tsv), [complete Slurm report](report.json)
and [response conventions](../../CONVENTIONS.md#6-response-estimators).
This completes the frozen-factor diagnostic. No model was selected or retrained.
To determine the *absolute* primary and neighbour errors, the next discriminating
measurement is the [proposed separate-shear image replay](../primary_neighbour_grid_20261106/design.md#four-shear-corners-and-exact-closure);
the present catalogues provide coherent response, not both of those physical labels.

## Why this comparison

The previous moment constraint improved selected-neighbour association in
both the flow's own data and the four constant-shear scenes. Nevertheless,
the flow contribution fell by more than the neighbour contribution rose;
all nine nearby cuts shifted toward more positive joint bias. See the
[completed comparison](../neighbour_moment_guard_20261106/results.md).

The flow contribution can change in two ways. Generated size and magnitude
change selection and the photometry passed into the conditional shape model.
The learned conditional shape model itself can also change during the
combined NLL/response optimization, even though the additional neighbour
loss has no direct gradient in that factor. The cross-products separately
hold each factor fixed and measure the interaction between these changes.
They retain the original classifier and emulator.

This is a localization experiment. Internal cancellations remain acceptable;
the aim is to understand how to retain response robustness near the owner's
nine cuts. A hybrid that happens to agree on four development scenes is not
automatically a validated or promoted model.

## Execution history

Three original source products and four cross-products are fixed in
manifest.json. No optimizer updates are made. The current preparation/pilot
job is17810892, followed by three persistent workers17810893 and
reducer17810894. See [execution and gates](rule.md#correctness-gates).

Two preparation attempts stopped before constant-shear evaluation: the
first used joint-minus-photometry density to check a conditional factor,
which exposed float32 log-determinant addition roundoff; the second reused
stale imported bytecode on the compute node. The corrected check evaluates
the conditional factor directly and reconstructs the joint density including
its known arithmetic. The launcher now compiles checksum-verified source
bytes; the current job prints the expected preparation digest. Scientific
model arithmetic and all prior results are unchanged.

All seven products now pass preparation. Standardizers/outer transforms,
identity reconstruction, photometry-factor equality and save/reload checks
pass. Direct conditional-shape density differences are exactly zero. The
joint-density arithmetic reconstruction passes; the largest explicitly
computed float32 log-determinant addition roundoff is9.54e-7. All three
source endpoints reproduce the original NLL and response probes.

At the first scheduled follow-up (2026-11-07 01:59 CET), the pilot and two
workers have completed. Seven of eight case/axis evaluations pass all gates;
the remaining case1080/axis2 is progressing normally in worker17810893_1.
The dependent report17810894 awaits that case. The largest unchanged-source
response replay difference so far is1.97e-12; shared-photometry selection
masses and reference-score sums remain exactly equal. The two-hour
same-thread loop remains active and the older broad evaluation remains held.

## Own-data factor decomposition

The original response-regularization shift is predominantly a conditional
shape-factor change. At the central cut on training probes, the R11 change
is-0.011611: changing only photometry contributes-0.000359, changing only
conditional shapes contributes-0.011264, and their interaction is+0.000013.
R22 shows the same pattern; validation closely agrees. Roughly97% of these
diagonal response changes come from the conditional shape change.

For the added neighbour moment, training R11 changes by-0.001808, decomposed
into-0.000828 from photometry, -0.000984 from conditional shapes and+0.000004
interaction. R22 changes by-0.003678, with contributions-0.000795/-0.002885/
+0.000002. The stronger R22 movement is mainly in the learned shape factor.
These are fixed common-random-number probe differences without case SEs;
they are not the joint constant-shear calibration result. Full matrices and
all nine cuts are in `own_probe_factorial.json`; see
[the probe table](own_probe_results.md).

The added loss has zero direct shape-factor gradient. Indirect routes remain:
the response gradient depends on generated photometry, and global gradient
clipping couples the updates. Both previous training arms clipped all64
updates at norm5. This establishes possible routes, not their individual
causal contributions; no clipping setting or training recipe is changed here.

## Four-scene axis1 interim localization

All four axis1 cases are complete. The lightweight [interim reducer](interim.py)
reads only saved aggregate sums; [interim.md](interim.md) gives the central-cut
tables and `interim.json` retains all modes and nine-cut factorial contrasts.
The three unchanged sources reproduce the previous point estimates and
jackknife SEs across all14 saved cuts:1512 entries, maximum difference
3.40e-11. Factorial sums, provenance, checksums and simulation replay pass.
The full two-axis report remains pending; these are axis1 results only.

At radius>3px and magnitude<25, with actual usability and generated
photometry/selection, the original nine-cut response regularization changes
the joint residual DeltaR=Rsim-Rmodel by+0.012049 ±0.000074. Changing just
the conditional-shape factor contributes+0.011512 ±0.000022 (about96%);
photometry at the old shape factor contributes+0.000537 ±0.000064, with
negligible interaction. The model's flow contribution changes-0.011165,
close to the own-data training-probe change-0.011611. This directly localizes
the regularization-induced shift to the learned conditional-shape factor;
it does not by itself identify the true standalone component errors in
constant-shear scenes.

The conclusion survives the nine nearby cuts and both usability choices.
With actual usability/generated selection, the shape-only residual change
ranges+0.011184 to+0.011635, compared with total+0.011572 to+0.012491.
Predicted usability gives nearly the same ranges. Even with measured
selection, the central shape-only shift is+0.011236 ±0.000024 and the total
is+0.011755 ±0.000022. Generated photometry still enters conditional shapes
and the emulator in that measured-selection mode.

For the added selected-neighbour moment, changing only photometry increases
the emulator contribution by+0.002483 ±0.000015, but decreases the flow
contribution by-0.002268 ±0.000083. Their net residual change is only
-0.000215 ±0.000088. Changing only the shape factor instead increases the
residual by+0.000874 ±0.000016; interaction is-0.000007 ±0.000002. Together
the residual increases+0.000652 ±0.000097. Thus much of the photometry gain
cancels within the fixed old conditional-shape model before the learned
shape factor's additional change is applied.

The photometry-only effect is not uniformly favorable: across the nine cuts
its residual change spans-0.000489 to+0.000402 with actual usability, and
-0.000545 to+0.000225 with predicted usability. The shape-only effect is
positive at every nearby cut in both modes. Measured selection also retains
this positive shape-factor shift (central+0.000827 ±0.000015). Improving
selected-neighbour association is therefore not sufficient to improve shear
response, and simply freezing shapes is not yet a robust solution.

All quoted SEs here are paired whole-case delete-one jackknife estimates,
conditional on fixed models and common draws. The much larger absolute
calibration uncertainties must not be replaced by these paired-change SEs:
central actual-usability m1 is-0.2840% ±0.6943pp initially, +1.0564% ±0.6954pp
after response regularization, and+1.1300% ±0.7035pp with the added moment.
Only four reused development scenes are represented. No endpoint is promoted
or selected. Next: check the final axis2 receipt and full report, then compare
the two axes before deciding whether a further focused diagnostic is justified.

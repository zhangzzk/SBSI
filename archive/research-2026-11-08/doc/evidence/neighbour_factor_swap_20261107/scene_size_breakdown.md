# Scene and intrinsic-size comparison

Four reused development scenes; central cut is FLUX_RADIUS > 3 pixels and MAG_AUTO < 25. Errors below are delete-one-scene jackknife SEs, conditional on fixed models and common draws. Differences between axes use the same scene deletion on both axes. The pooled estimate uses leg sums rather than an unweighted mean of the four single-scene ratios.

The [nine-cut table](cut_contrasts.tsv) gives every axis, matched axis difference, flag mode, factor contrast, flow/emulator change, residual change and individual-scene residual change. A positive residual shift means the model response fell: ΔR = simulation − model. Components are changes in predictions.

## Central residual and component changes

| Comparison | Axis | Photo-only residual | Shape-only residual | Interaction | Total residual | Total model flow change | Total model emulator change |
|---|---|---:|---:|---:|---:|---:|---:|
| response_regularization | 1 | +0.000537 ± 0.000064 | +0.011512 ± 0.000022 | -0.000001 ± 0.000003 | +0.012049 ± 0.000074 | -0.011165 ± 0.000071 | -0.000885 ± 0.000003 |
| response_regularization | 2 | +0.000537 ± 0.000012 | +0.008303 ± 0.000024 | -0.000004 ± 0.000004 | +0.008836 ± 0.000015 | -0.008188 ± 0.000015 | -0.000648 ± 0.000002 |
| response_regularization | 22-11 | +0.000000 ± 0.000070 | -0.003210 ± 0.000031 | -0.000003 ± 0.000006 | -0.003213 ± 0.000064 | +0.002976 ± 0.000061 | +0.000236 ± 0.000003 |
| neighbour_moment | 1 | -0.000215 ± 0.000088 | +0.000874 ± 0.000016 | -0.000007 ± 0.000002 | +0.000652 ± 0.000097 | -0.003246 ± 0.000090 | +0.002594 ± 0.000015 |
| neighbour_moment | 2 | -0.000239 ± 0.000090 | +0.003089 ± 0.000013 | -0.000010 ± 0.000005 | +0.002840 ± 0.000084 | -0.005219 ± 0.000077 | +0.002379 ± 0.000012 |
| neighbour_moment | 22-11 | -0.000024 ± 0.000160 | +0.002215 ± 0.000021 | -0.000003 ± 0.000003 | +0.002188 ± 0.000155 | -0.001972 ± 0.000151 | -0.000215 ± 0.000005 |

## Flag choices

| Comparison | Usability / selection | Total ΔR11 shift | Total ΔR22 shift | Shape-only ΔR11 shift | Shape-only ΔR22 shift |
|---|---|---:|---:|---:|---:|
| response_regularization | predicted / generated | +0.012018 ± 0.000051 | +0.008836 ± 0.000032 | +0.011508 ± 0.000023 | +0.008302 ± 0.000025 |
| response_regularization | actual / generated | +0.012049 ± 0.000074 | +0.008836 ± 0.000015 | +0.011512 ± 0.000022 | +0.008303 ± 0.000024 |
| response_regularization | actual / measured | +0.011755 ± 0.000022 | +0.008704 ± 0.000031 | +0.011236 ± 0.000024 | +0.008190 ± 0.000034 |
| neighbour_moment | predicted / generated | +0.000549 ± 0.000091 | +0.002745 ± 0.000076 | +0.000872 ± 0.000016 | +0.003090 ± 0.000013 |
| neighbour_moment | actual / generated | +0.000652 ± 0.000097 | +0.002840 ± 0.000084 | +0.000874 ± 0.000016 | +0.003089 ± 0.000013 |
| neighbour_moment | actual / measured | +0.000734 ± 0.000015 | +0.002969 ± 0.000017 | +0.000827 ± 0.000015 | +0.003111 ± 0.000011 |

Measured selection uses actual usability and measured per-leg cuts, but retains generated inputs to the shape and emulator predictions.

## Scene-by-scene shape contrast

Actual usability, generated selection, central cut. No independent case SE is assigned to a single scene.

| Scene | Regularization: shape-only ΔR11 | ΔR22 | Added moment: shape-only ΔR11 | ΔR22 | Added moment: total ΔR22 − ΔR11 shift |
|---|---:|---:|---:|---:|---:|
| 1000 | +0.011553 | +0.008254 | +0.000861 | +0.003120 | +0.001803 |
| 1040 | +0.011469 | +0.008315 | +0.000916 | +0.003074 | +0.002073 |
| 1080 | +0.011548 | +0.008363 | +0.000877 | +0.003100 | +0.002448 |
| 1120 | +0.011479 | +0.008279 | +0.000843 | +0.003062 | +0.002426 |

## Intrinsic size contribution

Actual usability, generated selection, central cut. Each bin contributes to the selected population response using the full selected mass on its own shear leg. Rows add to the pooled change; they are neither within-bin responses nor evidence that size causes the shift. Bins use the existing circularized intrinsic radius (arcsec), distinct from the measured FLUX_RADIUS cut.

| Intrinsic radius | Regularization shape-only ΔR11 | ΔR22 | Added moment shape-only ΔR11 | ΔR22 | Added moment shape-only ΔR22 − ΔR11 |
|---|---:|---:|---:|---:|---:|
| < 0.15 | +0.000802 ± 0.000009 | +0.000546 ± 0.000005 | +0.000028 ± 0.000003 | +0.000161 ± 0.000003 | +0.000132 ± 0.000005 |
| [0.15, 0.25) | +0.005151 ± 0.000019 | +0.003703 ± 0.000016 | +0.000370 ± 0.000016 | +0.001080 ± 0.000013 | +0.000709 ± 0.000023 |
| [0.25, 0.4) | +0.004723 ± 0.000019 | +0.003264 ± 0.000012 | +0.000408 ± 0.000004 | +0.001532 ± 0.000009 | +0.001124 ± 0.000010 |
| [0.4, 0.7) | +0.000796 ± 0.000007 | +0.000668 ± 0.000003 | +0.000069 ± 0.000001 | +0.000313 ± 0.000001 | +0.000244 ± 0.000001 |
| ≥ 0.7 | +0.000040 ± 0.000002 | +0.000121 ± 0.000001 | -0.000002 ± 0.000000 | +0.000003 ± 0.000001 | +0.000005 ± 0.000001 |

For the added moment, the 0.15–0.40 arcsec bins contribute +0.001833 ± 0.000021 of the shape-only difference between axes (82.8% ± 0.3 percentage points). Their shares of the selected population at fixed control photometry, averaging normalized plus/minus legs, are 75.7% for axis1 and 75.7% for axis2.

## Checks and selected-neighbour proxy

All 1296 bin-summed pooled contrasts and their jackknife SEs match the Slurm report to 1e-11. Maximum separate-versus-pooled emulator roundoff: 8.88e-16. All 56 files match the recorded hashes. Matching totals across the 16 scene/axis/legs (not unique galaxies): 0 crossmatch rows without shape, 22180 shape rows without crossmatch, and 9470549 parent rows absent from the measurement crossmatch. Unmatched measurements do not enter measured selection; full truth parents remain available for predicted usability.

On the fixed common-four-usable cohort (axis1 split/coherent catalogues), measured-minus-generated selected reference-neighbour coefficient changes from +0.003112 ± 0.000094 to +0.000866 ± 0.000099; paired change -0.002246 ± 0.000017. This is a selected-population proxy; it contains no shear finite difference. Shared photometry gives exactly the same proxy regardless of conditional-shape factor.

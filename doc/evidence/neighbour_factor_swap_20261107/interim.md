# Completed-axis interim factor comparison

Only fully completed axes, all four development cases. Whole-case delete-one jackknife SE; fixed models and draws. DeltaR=simulation-model; m=100(Rsim/Rmodel-1). Component contrasts are model changes, not standalone errors.

Complete axes: [1]. Pending case(s) by axis: {'2': [1080]}.

Target: radius>3px and magnitude<25. Actual usability with generated photometry/selection.

Axis 1:

| Product | m (%) ± jackknife SE (pp) |
|---|---:|
| initial | -0.284034 ± 0.694325 |
| control | +1.056418 ± 0.695375 |
| weighted | +1.129999 ± 0.703513 |
| photo_control_shape_initial | -0.225007 ± 0.689122 |
| photo_initial_shape_control | +0.995923 ± 0.700406 |
| photo_weighted_shape_control | +1.032177 ± 0.701368 |
| photo_control_shape_weighted | +1.155082 ± 0.697434 |

| Contrast / contribution | Photo at old shapes | Shapes at old photo | Interaction | Total |
|---|---:|---:|---:|---:|
| response_regularization / model_flow_response | +0.000225 ± 0.000061 | -0.011389 ± 0.000022 | +0.000000 ± 0.000003 | -0.011165 ± 0.000071 |
| response_regularization / model_emulator_response | -0.000762 ± 0.000004 | -0.000123 ± 0.000000 | +0.000000 ± 0.000000 | -0.000885 ± 0.000003 |
| response_regularization / delta_response | +0.000537 ± 0.000064 | +0.011512 ± 0.000022 | -0.000001 ± 0.000003 | +0.012049 ± 0.000074 |
| neighbour_moment / model_flow_response | -0.002268 ± 0.000083 | -0.000983 ± 0.000016 | +0.000005 ± 0.000002 | -0.003246 ± 0.000090 |
| neighbour_moment / model_emulator_response | +0.002483 ± 0.000015 | +0.000109 ± 0.000001 | +0.000002 ± 0.000000 | +0.002594 ± 0.000015 |
| neighbour_moment / delta_response | -0.000215 ± 0.000088 | +0.000874 ± 0.000016 | -0.000007 ± 0.000002 | +0.000652 ± 0.000097 |


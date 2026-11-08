# Frozen-factor own-data response probes

Fixed common-random-number own-data probes; changes in model response, without case uncertainty. These are not joint constant-shear residuals.

Target: radius>3 pixels and magnitude<25. Each row is a change in model response.

| Comparison | Split | Component | Photometry at old shapes | Shapes at old photometry | Interaction | Total |
|---|---|---|---:|---:|---:|---:|
| response_regularization | train | R11 | -0.000359 | -0.011264 | +0.000013 | -0.011611 |
| response_regularization | train | R22 | -0.000252 | -0.008445 | -0.000000 | -0.008697 |
| response_regularization | validation | R11 | -0.000363 | -0.011325 | +0.000008 | -0.011680 |
| response_regularization | validation | R22 | -0.000229 | -0.008530 | -0.000013 | -0.008773 |
| neighbour_moment | train | R11 | -0.000828 | -0.000984 | +0.000004 | -0.001808 |
| neighbour_moment | train | R22 | -0.000795 | -0.002885 | +0.000002 | -0.003678 |
| neighbour_moment | validation | R11 | -0.000803 | -0.000928 | -0.000004 | -0.001735 |
| neighbour_moment | validation | R22 | -0.000837 | -0.002964 | +0.000002 | -0.003798 |

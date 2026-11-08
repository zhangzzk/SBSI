# Current-model measured-input test

Exact measured photometry and membership, actual usability per leg. The conditional shapes and emulator are shared by both frozen-photometry endpoints. This is a counterfactual, not a causal additive error budget.

Central radius >3 pixels, magnitude <25. Errors are whole-scene jackknife SE (percentage points).

| Cohort | Prediction | m1 (%) | m2 (%) | Half component difference (pp) |
|---|---|---:|---:|---:|
| original_four | control_true_usability_generated_selection | +0.9728 ± 0.7041 | +1.2865 ± 0.4830 | -0.1568 ± 0.2080 |
| original_four | weighted_true_usability_generated_selection | +0.8747 ± 0.6993 | +1.1727 ± 0.4805 | -0.1490 ± 0.2001 |
| original_four | measured_shared | -0.1086 ± 0.6625 | +0.3672 ± 0.3819 | -0.2379 ± 0.2219 |
| additional_thirty_six | control_true_usability_generated_selection | +1.1409 ± 0.3396 | -0.1931 ± 0.2597 | +0.6670 ± 0.2154 |
| additional_thirty_six | weighted_true_usability_generated_selection | +1.0201 ± 0.3397 | -0.3075 ± 0.2596 | +0.6638 ± 0.2154 |
| additional_thirty_six | measured_shared | +0.1420 ± 0.3422 | -1.1022 ± 0.2538 | +0.6221 ± 0.2158 |
| all_forty | control_true_usability_generated_selection | +1.1241 ± 0.3115 | -0.0453 ± 0.2476 | +0.5847 ± 0.1984 |
| all_forty | weighted_true_usability_generated_selection | +1.0055 ± 0.3116 | -0.1597 ± 0.2475 | +0.5826 ± 0.1983 |
| all_forty | measured_shared | +0.1169 ± 0.3133 | -0.9555 ± 0.2411 | +0.5362 ± 0.1993 |

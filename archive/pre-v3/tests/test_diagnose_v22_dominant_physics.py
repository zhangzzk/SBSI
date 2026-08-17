import numpy as np
import pandas as pd

from scripts.diagnose_v22_dominant_physics import (
    casewise_slopes,
    controlled_tail_contrast,
    derive_physical_features,
    holm_adjust,
)


def test_derive_physical_features_exact_relations():
    frame = pd.DataFrame({
        "dominant_flux_ratio_primary": [100.0],
        "primary_size": [1.0],
        "dominant_secondary_size": [2.0],
        "dominant_distance": [3.0],
        "dominant_abs_response": [0.1],
        "dominant_to_runner_up_abs_response": [10.0],
    })
    got = derive_physical_features(frame).iloc[0]
    assert np.isclose(got.log10_flux_ratio, 2.0)
    assert np.isclose(got.log10_size_ratio, np.log10(2.0))
    assert np.isclose(got.log10_overlap_scale, 0.0)
    assert np.isclose(
        got.log10_surface_brightness_ratio,
        2.0 - 2.0 * np.log10(2.0),
    )
    assert np.isclose(got.log10_dominant_abs_response, -1.0)
    assert np.isclose(got.log10_dominance_strength, 1.0)


def test_holm_adjust_is_monotone_in_sorted_order():
    got = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert np.isclose(got["a"], 0.03)
    assert np.isclose(got["b"], 0.06)
    assert np.isclose(got["c"], 0.20)


def _one_cell_design(feature="log10_flux_ratio"):
    return {
        "n_control_bins": 1,
        "control_edges": {
            "log10_dominant_abs_response": [],
            "log10_dominance_strength": [],
        },
        "feature_moments": {feature: {"mean": 0.0, "sd": 1.0}},
        "cell_weights": {"0": 1.0},
        "cell_feature_tails": {
            "0": {feature: {"q25": -1.0, "q75": 1.0}}
        },
    }


def test_controlled_tail_contrast_uses_case_as_unit():
    rows = []
    for case in range(1, 13):
        offset = 3.0 * case - 10.0
        for x in (-2.0, -1.0, 1.0, 2.0):
            rows.append({
                "case": case,
                "deficit": offset + 2.0 * x,
                "log10_flux_ratio": x,
                "log10_dominant_abs_response": 0.0,
                "log10_dominance_strength": 0.0,
            })
    got = controlled_tail_contrast(
        pd.DataFrame(rows), "deficit", "log10_flux_ratio",
        _one_cell_design(), min_group_rows=2,
    )
    assert got["high_minus_low"]["n_cases"] == 12
    assert np.isclose(got["high_minus_low"]["mean"], 6.0)


def test_casewise_slopes_remove_case_and_control_cell_offsets():
    rows = []
    for case in range(1, 13):
        case_offset = 2.5 * case - 8.0
        for cell, cell_offset in ((0, 20.0), (1, -10.0)):
            for x in (-2.0, -1.0, 1.0, 2.0):
                rows.append({
                    "case": case,
                    "deficit": case_offset + cell_offset + 2.0 * x,
                    "log10_flux_ratio": x,
                    "log10_dominant_abs_response": float(cell),
                    "log10_dominance_strength": 0.0,
                })
    design = _one_cell_design()
    design["n_control_bins"] = 2
    design["control_edges"]["log10_dominant_abs_response"] = [0.5]
    design["control_edges"]["log10_dominance_strength"] = [1.0]
    got = casewise_slopes(
        pd.DataFrame(rows), "deficit", ["log10_flux_ratio"], design
    )
    assert got["n_cases_fit"] == 12
    assert np.isclose(got["features"]["log10_flux_ratio"]["mean"], 2.0)

import pandas as pd
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("extract_scene_catalogue.py")


def _paired(conflict=False):
    truth = {
        "RA": [10.0, 10.1, 10.2],
        "DEC": [0.0, 0.0, 0.0],
        "redshift": [0.4, 0.5, 0.6],
        "r": [23.0, 26.5, 24.0],
        "Re": [0.7, 0.3, 0.8],
        "sersic_n": [1.0, 2.0, 3.0],
        "axis_ratio": [0.8, 0.7, 0.6],
        "position_angle": [0.0, 0.2, 0.4],
    }
    rows = []
    for primary, secondary in ((0, 1), (2, 1)):
        row = {"case": 0, "input_index": primary, "input_index_sec": secondary}
        for name, values in truth.items():
            row[f"{name}_input_p"] = values[primary]
            row[f"{name}_input_s"] = values[secondary]
        rows.append(row)
    if conflict:
        rows[1]["r_input_s"] = 25.5
    return pd.DataFrame(rows)


def test_extract_unions_primary_and_secondary_and_keeps_zero_weight_neighbours():
    scene, report = MODULE.extract_scene_catalogue(
        _paired(),
        cases=(0,),
        primary_mag_min=18.0,
        primary_mag_max=25.8,
        primary_re_min=0.5,
        primary_re_max=1.5,
    )
    assert set(scene["input_index"]) == {0, 1, 2}
    weights = scene.set_index("input_index")["prior_weight"]
    assert weights[0] == weights[2] == 1.0
    assert weights[1] == 0.0
    assert report["n_positive_prior_atoms"] == 2
    assert report["n_zero_weight_neighbours"] == 1


def test_extract_rejects_conflicting_truth_for_one_input_galaxy():
    with pytest.raises(ValueError, match="conflicting truth records"):
        MODULE.extract_scene_catalogue(
            _paired(conflict=True),
            cases=(0,),
            primary_mag_min=18.0,
            primary_mag_max=25.8,
            primary_re_min=0.5,
            primary_re_max=1.5,
        )

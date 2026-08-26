import pandas as pd
import pytest

from _script_loader import load_script_module


MODULE = load_script_module("assemble_fs2_scene_catalogue.py")


def _case(case, *, g1=0.0):
    return pd.DataFrame(
        {
            "index": [0, 1],
            "cata_idx": [case * 10, case * 10 + 1],
            "RA": [10.0, 10.001],
            "DEC": [0.0, 0.001],
            "g1": [g1, g1],
            "g2": [0.0, 0.0],
            "shear_component_convention": ["sky_cos_sin", "sky_cos_sin"],
            "position_angle": [10.0, 30.0],
            "redshift": [0.4, 0.8],
            "Re": [0.7, 0.3],
            "axis_ratio": [0.8, 0.6],
            "sersic_n": [1.0, 2.0],
            "r": [23.0, 26.0],
        }
    )


def test_assemble_fs2_retains_full_cases_and_assigns_only_primary_mass(tmp_path):
    paths = []
    for case in (0, 1):
        path = tmp_path / f"gals{case}_0.0.feather"
        _case(case).to_feather(path)
        paths.append(path)
    scene, reports = MODULE.assemble_fs2_scene_catalogue(
        paths,
        cases=(0, 1),
        expected_g1=0.0,
        expected_g2=0.0,
        primary_mag_bounds=(18.0, 25.8),
        primary_re_bounds=(0.5, 1.5),
    )
    assert len(scene) == 4
    assert scene.groupby("case")["prior_weight"].sum().to_dict() == {0: 1.0, 1: 1.0}
    assert [report["n_positive_prior_atoms"] for report in reports] == [1, 1]


def test_assemble_fs2_rejects_a_nonnull_source_case(tmp_path):
    path = tmp_path / "gals0_0.0.feather"
    _case(0, g1=0.02).to_feather(path)
    with pytest.raises(ValueError, match="declared shear"):
        MODULE.assemble_fs2_scene_catalogue(
            [path],
            cases=(0,),
            expected_g1=0.0,
            expected_g2=0.0,
            primary_mag_bounds=(18.0, 25.8),
            primary_re_bounds=(0.5, 1.5),
        )

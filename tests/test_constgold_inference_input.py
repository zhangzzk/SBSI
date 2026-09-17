import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_constgold_input", Path(__file__).resolve().parents[1] / "scripts/prepare_constgold_inference_input.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_raw_plus_selection_reports_unmatched_rows_and_keeps_large_shapes(tmp_path):
    root = tmp_path / "case40_0.02/real0/catalogues"
    for directory in ("input", "CrossMatch", "Shapes"):
        (root / directory).mkdir(parents=True)
    pd.DataFrame({
        "index_input": [1, 2, 3, 4, 5, 6], "r_input": [22.0] * 6,
        "Re_input": [0.8] * 6, "gamma1_input": [0.02] * 6, "gamma2_input": [0.0] * 6,
    }).to_feather(root / "input/gals_info_tile180.0_-0.5.feather")
    pd.DataFrame({"id_input": [1, 2, 3, 4, 5, 6, 99], "id_detec": [11, 12, 13, 14, 15, 16, 19]}).to_feather(
        root / "CrossMatch/tile180.0_-0.5_rot0_matched.feather"
    )
    shapes = pd.DataFrame({
        "NUMBER": [11, 12, 13, 14, 15, 19, 20],
        "NGMIX_G1": [0.9, -1.0, 0.0, 0.2, 0.3, 0.2, 0.2], "NGMIX_G2": [0.0] * 7,
        "MAG_AUTO": [24.0, 24.0, 24.0, 25.8, 24.0, 24.0, 24.0],
        "FLUX_RADIUS": [3.75, 4.0, 4.0, 4.0, 3.74, 4.0, 4.0],
    })
    shape_path = root / "Shapes/shape_catalogue_detect_position_all_tile180.0_-0.5.feather"
    shapes.to_feather(shape_path)
    args = dict(pixel_size=0.2, zero_point=30.0, primary_mag=(18, 25.8), primary_re=(0.5, 1.5))
    result, report = module.read_case(tmp_path, 40, **args)
    assert result.source_input_index.tolist() == [1]
    assert report["crossmatch_without_shape"] == 1
    assert report["shape_without_crossmatch"] == 1
    assert report["matched_without_truth"] == 1
    assert report["invalid_or_unusable"] == 2
    assert result.measured_ngmix_g1.iloc[0] == 0.9
    assert result.measured_flux_from_mag_auto.iloc[0] == pytest.approx(10**2.4)
    assert np.isfinite(result.to_numpy()).all()
    relaxed, _ = module.read_case(tmp_path, 40, measured_radius_min=0.5, **args)
    assert relaxed.source_input_index.tolist() == [1, 5]
    pd.concat([shapes, shapes.iloc[:1]], ignore_index=True).to_feather(shape_path)
    with pytest.raises(ValueError, match="duplicate NUMBER"):
        module.read_case(tmp_path, 40, **args)


def test_configured_selection_uses_pixel_units_and_rejects_unsupported_cuts():
    config = {
        "observing_conditions": {"pixel_size": 0.2, "zero_point": 30.0},
        "measured_selection": {
            "mode": "output_cut_with_population_normalization",
            "bounds": ["measured_flux_radius:2.5:", "measured_flux_from_mag_auto:47.8630092322638:"],
        },
    }
    assert module.selection_thresholds(config) == pytest.approx((0.5, 25.8))
    config["measured_selection"]["bounds"].append("measured_ngmix_g1:-0.8:0.8")
    with pytest.raises(ValueError):
        module.selection_thresholds(config)
